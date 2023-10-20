// Copyright 2023 EdgeCortix Inc.

#ifndef VMEIOCTL_H
#define VMEIOCTL_H

#define PCIE_NODE_PREFIX        "sakura_pcie"

#define CTRL_MINOR          0
#define MEM_DDR0_MINOR      1
#define MEM_DDR1_MINOR      2
#define MAX_MINORS          (MEM_DDR1_MINOR + 1)

#define SIZE_4MB             0x0400000
#define SIZE_16MB            0x1000000

#define IOCTL_READ_REG              0xF001
#define IOCTL_WRITE_REG             0xF002

typedef struct _reg_ops {
  uint32_t        rw;
  uint32_t        value;
  uint64_t        reg_addr;
} reg_ops;


#define IOCTL_READ_MEM              0xF003
#define IOCTL_WRITE_MEM             0xF004

typedef struct _mem_ops {
  uint32_t        rw;
  uint32_t        size;
  uint32_t        type;
  uint32_t        timeout;
  uint64_t        mem_addr;
} mem_ops;


#endif
