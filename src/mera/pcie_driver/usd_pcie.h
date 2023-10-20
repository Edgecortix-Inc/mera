// Copyright 2023 EdgeCortix Inc.

#ifndef _USD_PCIE_H
#define _USD_PCIE_H

#define PCI_VENDOR_ID_CDNS_REV2_PCIE_EP  0x17CD
#define PCI_VENDOR_ID_CDNS_REV3_PCIE_EP  0x1FDC
#define PCI_DEVICE_ID_CDNS_PCIE_EP  0x0100

#define CONFIG_BAR      0
#define DDR0_P0_BAR     2
#define DDR1_P0_BAR     4
#define REG_LEN         4
#define SAKURA_PREFIX  "Sakura_PCIe: "

typedef struct _image_desc_t {
  void __iomem           *map_addr;
  unsigned int           size;
  struct device          *dev;
  atomic_t               refcnt;
  atomic_t               map_enabled;
  char                   name[64];
} image_desc_t;


typedef struct _endpoint_dev {
  struct pci_dev          *pdev;

  void __iomem            *regaddr;
  uint64_t                reg_len;

  void __iomem            *memddr0_virt;
  uint64_t                memddr0_len;
  uint64_t                memddr0_phys;

  void __iomem            *memddr1_virt;
  uint64_t                memddr1_len;
  uint64_t                memddr1_phys;
  uint32_t                bar2_enabled;

  spinlock_t              reg_lock;
  struct mutex            mem_lock;

  struct cdev             cdev;
  struct class            *devclass;

  dev_t                   node;
  int                     major;
  int                     lowest_minor;
  unsigned int            num_nodes;

  unsigned int            num_windows;
  image_desc_t            image[MAX_MINORS];
} endpoint_dev_t;


//////// function 0 config registers
#define     PCIE_CFG_REG_BASE       0x40000

#define     PCI_CSR                 0x4
#define     PCI_CSR_BUS_ERRORS      0xf9000000

#define     PCI_CSR_MDPE            0x01000000
#define     PCI_CSR_STA             0x08000000
#define     PCI_CSR_RTA             0x10000000
#define     PCI_CSR_RMA             0x20000000
#define     PCI_CSR_SSE             0x40000000
#define     PCI_CSR_DPE             0x80000000

#endif
