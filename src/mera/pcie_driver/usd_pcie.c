// Copyright 2023 EdgeCortix Inc.

#include <linux/version.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/interrupt.h>
#include <linux/moduleparam.h>
#include <linux/spinlock.h>
#include <linux/sched.h>
#include <linux/poll.h>
#include <linux/proc_fs.h>
#include <linux/delay.h>
#include <linux/cdev.h>
#include <linux/ioctl.h>
#include <linux/mutex.h>

#include "pcieioctl.h"
#include "usd_pcie.h"

static const char pcie_name[] = PCIE_NODE_PREFIX;

static const struct pci_device_id pcie_ids[] = {
  {PCI_DEVICE(PCI_VENDOR_ID_CDNS_REV2_PCIE_EP, PCI_DEVICE_ID_CDNS_PCIE_EP)},
  {PCI_DEVICE(PCI_VENDOR_ID_CDNS_REV3_PCIE_EP, PCI_DEVICE_ID_CDNS_PCIE_EP)},
  {0}
};

static void clear_bus_err(endpoint_dev_t *epdev) {
  u32 status = 0;
  uint64_t reg;
  spin_lock(&epdev->reg_lock);

  reg = PCIE_CFG_REG_BASE + PCI_CSR;
  status = readl(epdev->regaddr + reg);

  if (status & PCI_CSR_BUS_ERRORS) {
    printk(KERN_ERR SAKURA_PREFIX "Bus error, PCI_CSR: [0x%08x], clearing it !\n", status);
    writel(status, epdev->regaddr + reg);
  }
  spin_unlock(&epdev->reg_lock);
}

static ssize_t pcie_read(struct file *file, char __user *buf,
                         size_t count, loff_t *ppos) {
  unsigned int minor = MINOR(file->f_path.dentry->d_inode->i_rdev);
  printk(KERN_INFO SAKURA_PREFIX "minor[%d]; read not supported on this device\n", minor);
  return -EINVAL;
}

static ssize_t pcie_write(struct file *file, const char __user *buf,
                          size_t count, loff_t *ppos) {
  unsigned int minor = MINOR(file->f_path.dentry->d_inode->i_rdev);
  printk(KERN_INFO SAKURA_PREFIX "minor[%d]: write not supported on this device\n", minor);
  return -EINVAL;
}

static int pcie_mmap(struct file *file, struct vm_area_struct *vma) {
  unsigned int minor = MINOR(file->f_path.dentry->d_inode->i_rdev);
  struct inode *inode = file->f_path.dentry->d_inode;

  endpoint_dev_t *epdev = NULL;
  unsigned int  size = 0;
  int ret;

  if((minor == CTRL_MINOR) || (minor >= MAX_MINORS)) {
    printk(KERN_INFO SAKURA_PREFIX "minor[%d]: mmap: Invalid for this window\n", minor);
    return -EINVAL;
  }

  if((vma->vm_end - vma->vm_start) != SIZE_4MB) {
    printk(KERN_INFO SAKURA_PREFIX "minor[%d]: mmap: Invalid size for this window, setting it to 4MB\n", minor);
  }
  size = SIZE_4MB;

  epdev = container_of(inode->i_cdev, endpoint_dev_t, cdev);
  if(!epdev) {
    printk(KERN_INFO SAKURA_PREFIX "minor[%d]: Invalid device\n", minor);
    return -EINVAL;
  }

  if(atomic_read(&epdev->image[minor].map_enabled)) {
    printk(KERN_INFO SAKURA_PREFIX "minor[%d]: Mapping already enabled on this device\n", minor);
    return -EINVAL;
  }

  vma->vm_flags |= VM_LOCKED;

  // Calculate page frame number (PFN).
  vma->vm_pgoff = virt_to_phys(epdev->image[minor].map_addr) >> PAGE_SHIFT;
  ret = remap_pfn_range(vma, vma->vm_start, vma->vm_pgoff, size, vma->vm_page_prot);
  if (ret != 0) {
    printk(KERN_ERR SAKURA_PREFIX "minor[%d]: mmap: remap_pfn_range failed w/err [%d]!\n", minor, ret);
    return (-EAGAIN);
  }

  vma->vm_file = file;

  atomic_set(&epdev->image[minor].map_enabled, 1);
  printk(KERN_INFO SAKURA_PREFIX "minor[%d]: Mapping enabled\n", minor);
  return 0;
}

static int pcie_open(struct inode *inode, struct file *file) {
  unsigned int minor = MINOR(inode->i_rdev);
  endpoint_dev_t *epdev = NULL;

  if (minor >= MAX_MINORS) {
    printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid window\n", minor);
    return -ENODEV;
  }

  epdev = container_of(inode->i_cdev, endpoint_dev_t, cdev);
  if(!epdev) {
    printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid device\n", minor);
    return -EINVAL;
  }

  if((minor != CTRL_MINOR) && (atomic_read(&epdev->image[minor].refcnt) > 0)) {
    printk(KERN_INFO SAKURA_PREFIX "minor[%d]: Already opened\n", minor);
    return -EBUSY;
  }

  atomic_inc(&epdev->image[minor].refcnt);
  return 0;
}

static int pcie_release(struct inode *inode, struct file *file) {
  unsigned int minor = MINOR(inode->i_rdev);
  endpoint_dev_t *epdev = NULL;

  if (minor >= MAX_MINORS) {
    printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid window\n", minor);
    return -ENODEV;
  }

  epdev = container_of(inode->i_cdev, endpoint_dev_t, cdev);
  if(!epdev) {
    printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid device\n", minor);
    return -EINVAL;
  }

  atomic_dec(&epdev->image[minor].refcnt);
  atomic_set(&epdev->image[minor].map_enabled, 0);

  return 0;
}

int write_memory(endpoint_dev_t *epdev, unsigned int minor, mem_ops mops) {
  void __iomem * addr = 0;

  // set add in DDRx mem region
  if(minor == MEM_DDR0_MINOR) {
    addr = epdev->memddr0_virt + mops.mem_addr;
  } else {
    if(!epdev->bar2_enabled)
      return -ENODEV;
    addr = epdev->memddr1_virt + mops.mem_addr;
  }

  mutex_lock(&epdev->mem_lock);
  memcpy_toio((volatile void __iomem *)addr, (void *)epdev->image[minor].map_addr, mops.size);
  mutex_unlock(&epdev->mem_lock);
  clear_bus_err(epdev);
  return 0;
}

int read_memory(endpoint_dev_t *epdev, unsigned int minor, mem_ops mops) {
  void __iomem * addr = 0;
  uint8_t *dataptr = NULL;

  // set add in DDRx mem region
  if(minor == MEM_DDR0_MINOR) {
    addr = epdev->memddr0_virt + mops.mem_addr;
  } else {
    if(!epdev->bar2_enabled)
      return -ENODEV;
    addr = epdev->memddr1_virt + mops.mem_addr;
  }

  // data to be placed in image buffer
  dataptr = epdev->image[minor].map_addr;

  mutex_lock(&epdev->mem_lock);
  memcpy_fromio((void *)epdev->image[minor].map_addr, (const volatile void *)addr, mops.size);
  mutex_unlock(&epdev->mem_lock);
  return 0;
}

static long pcie_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {
  int err = 0;

  unsigned int minor = MINOR(file->f_path.dentry->d_inode->i_rdev);
  struct inode *inode = file->f_path.dentry->d_inode;
  endpoint_dev_t *epdev = NULL;

  if((cmd < IOCTL_READ_REG) || (cmd > IOCTL_WRITE_MEM)) {
    printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid ioctl %d\n", minor, cmd);
    return -EFAULT;
  }

  epdev = container_of(inode->i_cdev, endpoint_dev_t, cdev);
  if(!epdev) {
    printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid device\n", minor);
    return -EINVAL;
  }
  switch (cmd) {
  case IOCTL_READ_REG: 
  {
    reg_ops rops;
    if(minor != CTRL_MINOR) {
      printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid reg read on mem window\n", minor);
      return -EFAULT;
    }

    err = copy_from_user(&rops, (void *) arg, sizeof(reg_ops));
    if(err != 0) {
      printk(KERN_ERR SAKURA_PREFIX "minor[%d]:get reg details from lib failed w/err [%d]\n", minor, err);
      return -EFAULT;
    }

    if(rops.reg_addr >= (epdev->reg_len - REG_LEN)) {
      printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid reg addr [0x%llx]\n", minor, rops.reg_addr);
      return -EINVAL;
    }

    spin_lock(&epdev->reg_lock);
    rops.value = readl(epdev->regaddr + rops.reg_addr);
    spin_unlock(&epdev->reg_lock);                

    err = __copy_to_user((void *)arg, &rops, sizeof(reg_ops));
    if(err != 0) {
      printk(KERN_ERR SAKURA_PREFIX "minor[%d]: send reg val to lib failed w/err [%d]\n", minor, err);
      return -EFAULT;
    }
  }
  break;

  case IOCTL_WRITE_REG: 
  {
    reg_ops rops;
    if(minor != CTRL_MINOR) {
      printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid reg write on mem window\n", minor);
      return -EFAULT;
    }

    err = copy_from_user(&rops, (char *) arg, sizeof(reg_ops));
    if(err != 0) {
      printk(KERN_ERR SAKURA_PREFIX "minor[%d]: get reg details from lib failed w/err [%d]\n", minor, err);
      return -EFAULT;
    }

    if(rops.reg_addr >= (epdev->reg_len - REG_LEN)) {
      printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid reg addr [0x%llx]\n", minor, rops.reg_addr);
      return -EINVAL;
    }

    spin_lock(&epdev->reg_lock);
    writel(rops.value, epdev->regaddr + rops.reg_addr);
    spin_unlock(&epdev->reg_lock);
  }
  break;

  case IOCTL_READ_MEM: 
  {
    mem_ops mops;
    if(minor == CTRL_MINOR) {
      printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid mem read on control window\n", minor);
      return -EFAULT;
    }

    err = copy_from_user(&mops, (char *) arg, sizeof(mem_ops));
    if(err != 0) {
      printk(KERN_ERR SAKURA_PREFIX "minor[%d]: get mem details from lib failed w/err [%d]\n", minor, err);
      return -EFAULT;
    }

    if(mops.type != minor) {
      printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid mem node [%d]\n", minor, mops.type);
      return -EFAULT;
    }

    //check if axi addr is not overlapping scratchpad mem or out of bound of DDRx mem region;
    if(MEM_DDR0_MINOR) {
      if((mops.mem_addr + mops.size) > epdev->memddr0_len) {
        printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid addr, out of mapped area\n", minor);
        return -EINVAL;
      }
    } else {
      if((mops.mem_addr + mops.size) > epdev->memddr1_len) {
        printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid addr, out of mapped area\n", minor);
        return -EINVAL;
      }
    }
    return read_memory(epdev, minor, mops);
  }
  break;

  case IOCTL_WRITE_MEM: 
  {
    mem_ops mops;
    if(minor == CTRL_MINOR) {
      printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid mem write on control window\n", minor);
      return -EFAULT;
    }

    err = copy_from_user(&mops, (char *) arg, sizeof(mem_ops));
    if(err != 0) {
      printk(KERN_ERR SAKURA_PREFIX "minor[%d]: get mem details from lib failed w/err [%d]\n", minor, err);
      return -EFAULT;
    }

    if(mops.type != minor) {
      printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid mem node [%d]\n", minor, mops.type);
      return -EFAULT;
    }

    //check if axi addr is not overlapping scratchpad mem or out of bound of DDRx mem region;
    if(MEM_DDR0_MINOR) {
      if((mops.mem_addr + mops.size) > epdev->memddr0_len) {
        printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid addr, out of mapped area\n", minor);
        return -EINVAL;
      }
    } else {
      if((mops.mem_addr + mops.size) > epdev->memddr1_len) {
        printk(KERN_INFO SAKURA_PREFIX "minor[%d]: invalid addr, out of mapped area\n", minor);
        return -EINVAL;
      }
    }
    return write_memory(epdev, minor, mops);
  }
  break;
  default:
    return -ENOIOCTLCMD;
  }

  return 0;
}



static struct file_operations sakura_pcie_fops = {
  .owner          = THIS_MODULE,
  .open           = pcie_open,
  .release        = pcie_release,
  .read           = pcie_read,
  .write          = pcie_write,
  .unlocked_ioctl = pcie_ioctl,
  .mmap           = pcie_mmap
};

static int pcie_probe(struct pci_dev *pdev, const struct pci_device_id *ent) {
  int ret, i;
  endpoint_dev_t *epdev = devm_kzalloc(&pdev->dev, sizeof(endpoint_dev_t), GFP_KERNEL);

  printk(KERN_INFO SAKURA_PREFIX "Sakura device found.\n");

  if(epdev == NULL) {
    printk(KERN_ERR SAKURA_PREFIX "dev allocation failed\n");
    return -ENOMEM;
  }

  pci_set_drvdata(pdev, epdev);
  ret = pci_enable_device(pdev);
  if (ret) {
    printk(KERN_ERR SAKURA_PREFIX "Failed to enable Sakura device.\n");
    return ret;
  }

  printk(KERN_INFO SAKURA_PREFIX "Sakura device enabled.\n");

  if(!(pci_resource_flags(pdev, CONFIG_BAR) & IORESOURCE_MEM)) {
    printk(KERN_ERR SAKURA_PREFIX "Incorrect BAR0 configuration\n");
    ret = -ENODEV;
    goto disable_device;
  }
     
  if(!(pci_resource_flags(pdev, DDR0_P0_BAR) & IORESOURCE_MEM)) {
    printk(KERN_ERR SAKURA_PREFIX "Incorrect BAR1 configuration.\n");
    ret =  -ENODEV;
    goto disable_device;
  }
  printk(KERN_INFO SAKURA_PREFIX "found at bus %x device %x\n", pdev->bus->number, pdev->devfn);
  printk(KERN_INFO SAKURA_PREFIX "Vendor = %04X Device = %04X\n", pdev->vendor, pdev->device);
  printk(KERN_INFO SAKURA_PREFIX "Class = %08X\n", pdev->class);

  epdev->reg_len = pci_resource_len(pdev, CONFIG_BAR);
  epdev->regaddr = pci_iomap(pdev, CONFIG_BAR, epdev->reg_len);
  if (epdev->regaddr == NULL) {
    printk(KERN_ERR SAKURA_PREFIX "Request pci region failed.\n");
    ret = -ENOMEM;
    goto disable_device;
  }
  epdev->memddr0_len = pci_resource_len(pdev, DDR0_P0_BAR);
  epdev->memddr0_virt = pci_iomap(pdev, DDR0_P0_BAR, epdev->memddr0_len);
  if (epdev->memddr0_virt == NULL) {
    printk(KERN_ERR SAKURA_PREFIX "Request memddr0_virt failed.\n");
    ret = -ENOMEM;
    goto unmap_bar0;
  }
  epdev->memddr0_phys = pci_resource_start(pdev, DDR0_P0_BAR);

  epdev->memddr1_len = pci_resource_len(pdev, DDR1_P0_BAR);
  epdev->memddr1_virt = pci_iomap(pdev, DDR1_P0_BAR, epdev->memddr1_len);
  if (epdev->memddr1_virt == NULL) {
    printk(KERN_ERR SAKURA_PREFIX "Request memddr1_virt failed.\n");
    ret = -ENOMEM;
    epdev->bar2_enabled = 0;
    goto unmap_bar1;
  } else {
    epdev->bar2_enabled = 1;
  }
  epdev->memddr1_phys = pci_resource_start(pdev, DDR1_P0_BAR);

  pci_set_master(pdev);
  if (pcie_set_readrq(pdev, 4096)) {
    printk(KERN_ERR SAKURA_PREFIX "Can not update MaxReadReq\n");
  }

  if (dma_set_mask(&pdev->dev, DMA_BIT_MASK(64))) {
    printk(KERN_ERR SAKURA_PREFIX "Failed to set DMA mask. Aborting.\n");
    ret = -ENODEV;
    goto unmap_bar2;
  }
  epdev->num_nodes = MAX_MINORS;

  ret = alloc_chrdev_region(&epdev->node, 0, epdev->num_nodes, pcie_name);
  if(ret) {
    printk(KERN_ERR SAKURA_PREFIX "Failed to alloc chrdev region\n");
    ret = -ENODEV;
    goto unmap_bar2;
  }

  epdev->major = MAJOR(epdev->node);
  epdev->lowest_minor = MINOR(epdev->node);

  cdev_init(&epdev->cdev, &sakura_pcie_fops);

  epdev->cdev.owner = THIS_MODULE;

  ret = cdev_add(&epdev->cdev, MKDEV(epdev->major, epdev->lowest_minor), epdev->num_nodes);
  if (ret) {
    printk(KERN_ERR SAKURA_PREFIX "Failed to add cdev. Aborting.\n");
    goto err_unregister_chrdev;
  }

  epdev->devclass = class_create(THIS_MODULE, pcie_name);
  if (IS_ERR(epdev->devclass)) {
    printk(KERN_ERR SAKURA_PREFIX "Failed to create class for cdev. Aborting.\n");
    ret = -ENODEV;
    goto err_cdev_del;
  }


  for (i = epdev->lowest_minor; i < epdev->num_nodes; i++) {
    snprintf(epdev->image[i].name, sizeof(epdev->image[i].name) - 1, "%s_%d", PCIE_NODE_PREFIX, i);
    epdev->image[i].name[sizeof(epdev->image[i].name) - 1] = '\0';

    epdev->image[i].dev = device_create(epdev->devclass, NULL, MKDEV(epdev->major, i), NULL, epdev->image[i].name);

    if (IS_ERR(epdev->image[i].dev)) {
      printk(KERN_ERR SAKURA_PREFIX "Failed to create %s device. Aborting.\n", epdev->image[i].name);
      ret = -ENODEV;
      goto unroll_device_create;
    }
    printk(KERN_INFO SAKURA_PREFIX "[%s]: Created device file\n", epdev->image[i].name);
  }
  epdev->num_windows = MAX_MINORS - 1;

  for (i = 1; i <= epdev->num_windows; i++) {
    epdev->image[i].map_addr = kmalloc(SIZE_4MB, GFP_KERNEL);
    if (epdev->image[i].map_addr == NULL) {
      printk(KERN_ERR SAKURA_PREFIX "[%s]: Unable to allocate memory!\n", epdev->image[i].name);
      ret = -ENOMEM;
      goto unroll_mem_windows;
    } else {
      struct page *page = NULL;
      for (page = virt_to_page(epdev->image[i].map_addr);
           page < virt_to_page(epdev->image[i].map_addr + SIZE_4MB); ++page)
        SetPageReserved(page);
    }
  }

  // init spin locks
  mutex_init(&epdev->mem_lock);
  spin_lock_init(&epdev->reg_lock);
  return ret;

unroll_mem_windows:
  i--;
  for (; i > 0; i--) {
    struct page *page = NULL;
    for (page = virt_to_page(epdev->image[i].map_addr);
         page < virt_to_page(epdev->image[i].map_addr + SIZE_4MB);  ++page)
      ClearPageReserved(page);
    kfree(epdev->image[i].map_addr);
  }

unroll_device_create:
  i--;
  for (; i >= 0; i--) {
    device_destroy(epdev->devclass, MKDEV(epdev->major, i));
  }
  class_destroy(epdev->devclass);

err_cdev_del:
  cdev_del(&epdev->cdev);
err_unregister_chrdev:
  unregister_chrdev_region(MKDEV(epdev->major, epdev->lowest_minor), epdev->num_nodes);
unmap_bar2:
  if(epdev->memddr1_virt)
    pci_iounmap(pdev, epdev->memddr1_virt);
unmap_bar1:
  pci_iounmap(pdev, epdev->memddr0_virt);
unmap_bar0:
  pci_iounmap(pdev, epdev->regaddr);
disable_device:
  pci_disable_device(pdev);

  return ret;
}

static void pcie_remove(struct pci_dev *pdev) {
  int i;
  endpoint_dev_t *epdev = pci_get_drvdata(pdev);

  for (i = 1; i <= epdev->num_windows; i++) {
    struct page *page = NULL;
    for (page = virt_to_page(epdev->image[i].map_addr);
         page < virt_to_page(epdev->image[i].map_addr + SIZE_4MB);  ++page)
      ClearPageReserved(page);
    kfree(epdev->image[i].map_addr);
  }

  printk(KERN_INFO SAKURA_PREFIX "Removed %d mem windows.\n", epdev->num_windows);

  for (i = 0; i < epdev->num_nodes; i++)
    device_destroy(epdev->devclass, MKDEV(epdev->major, i));

  printk(KERN_INFO SAKURA_PREFIX "Removed %d device files.\n", epdev->num_nodes);
  class_destroy(epdev->devclass);
  cdev_del(&epdev->cdev);
  unregister_chrdev_region(MKDEV(epdev->major, epdev->lowest_minor), epdev->num_nodes);

  if(epdev->regaddr)
    pci_iounmap(pdev, epdev->regaddr);
  if(epdev->memddr0_virt)
    pci_iounmap(pdev, epdev->memddr0_virt);
  if(epdev->memddr1_virt)
    pci_iounmap(pdev, epdev->memddr1_virt);
}

MODULE_DEVICE_TABLE(pci, pcie_ids);

static struct pci_driver pcie_driver = {
  .name = pcie_name,
  .id_table = pcie_ids,
  .probe = pcie_probe,
  .remove = pcie_remove,
};


MODULE_DESCRIPTION("Sakura PCIe Target Driver");
MODULE_AUTHOR("Edgecortix Inc.");
MODULE_VERSION("0.2.0");
MODULE_ALIAS("pcie_sakura");
MODULE_LICENSE("GPL v2");

module_pci_driver(pcie_driver);


