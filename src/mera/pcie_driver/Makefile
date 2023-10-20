
obj-m 	:= pcie_sakura.o
pcie_sakura-objs := usd_pcie.o

RELEASE := $(shell uname -r)
KDIR ?= /lib/modules/$(RELEASE)/build
CMD := $(MAKE) -C $(KDIR) M=$(PWD)

all default: clean
	$(CMD) modules

clean:
	$(CMD) clean


