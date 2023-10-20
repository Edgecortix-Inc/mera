#! /usr/bin/env python3

import argparse
import mera
import sys
import subprocess
import os

from pathlib import Path

__mera_loc__ = Path(mera.__file__).resolve().parent

def cmd(command, cwd = os.getcwd(), print_stdout = True, assert_retcode = False):
    stdout_str = ''
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd) as p:
        for line in p.stdout:
            line_str = line.decode('utf-8')
            stdout_str += line_str
            if print_stdout:
                print(line_str, end='')
    if assert_retcode and int(p.returncode) != 0:
        raise ValueError(f'Failed to run command "{command}". Ret code is {int(p.returncode)}')
    return p.returncode, stdout_str

def find_bin_util(name):
    return str(__mera_loc__ / 'bin_utils' / name)

def check_for_driver(driver_name, supported_drivers) -> str:
    orig_name = driver_name
    uname = str(cmd('uname -r', print_stdout=False)[1]).strip()
    driver_name = f'{driver_name}_{uname}.ko'
    if not Path(find_bin_util(driver_name)).exists():
        print(f'ERROR: Cannot find suitable {orig_name} driver for Linux version "{uname}"\n'
            + f'Available driver(s): {supported_drivers}\nExiting...')
        return None
    return driver_name

def main():
    arg_p = argparse.ArgumentParser(description='Utility to query information about MERA platform')
    arg_p.add_argument('-v', '--version', action='store_true', help='Display the current version about the installed MERA')
    arg_p.add_argument('--intel_get_board_id', nargs='?', required=False, const='',
        help='On intel boards prints the current board id. Optionally provide the BDF location as argument')
    arg_p.add_argument('--intel_start_daemon', action='store_true', help='Setup and start intel MERA PCIe daemon', required=False)
    arg_p.add_argument('--sakura1_start', nargs='?', const='', help='Setups a SAKURA_1 machine after boot up.', required=False)

    args = arg_p.parse_args()
    if args.version:
        print(mera.get_versions())
        return 0
    elif args.intel_get_board_id is not None:
        return cmd(f'{find_bin_util("intel_get_board_id")} {args.intel_get_board_id}')[0]
    elif args.intel_start_daemon:
        # Check we have driver for that version available and we are running as root
        driver_name = check_for_driver("ifc_uio", ['5.15.0-56-generic'])
        if not driver_name:
            return -1
        # Load kernel driver
        cmd(f'sudo modprobe uio', assert_retcode=True)
        cmd(f'sudo insmod {find_bin_util(driver_name)} || true', print_stdout=False, assert_retcode=True)
        cmd(f'sudo chmod ugo+rwx /dev/hugepages/', assert_retcode=True)
        cmd(f'sudo chmod ugo+rwx /dev/uio0', assert_retcode=True)
        cmd(f'sudo chmod ugo+rwx /sys/class/uio/uio0/device/resource*', assert_retcode=True)
        # Start dma daemon in the background
        pid = subprocess.Popen(['sudo', find_bin_util("ec_dma_daemon_proc")], close_fds=True).pid
        print(f'Started DMA daemon with PID {pid}')
        return 0
    elif args.sakura1_start is not None:
        # Compile pcie kernel driver
        __DRIVER_NAME = 'pcie_sakura.ko'
        cmd(f'rm {__DRIVER_NAME}; make', cwd=__mera_loc__ / 'pcie_driver', print_stdout=False, assert_retcode=True)
        # Check it is correctly compiled
        __DRIVER_LOC = __mera_loc__ / 'pcie_driver' / __DRIVER_NAME
        if not __DRIVER_LOC.exists():
            print(f'ERROR: Failed to find SAKURA PCIe driver.')
            return -1
        # Load kernel driver
        cmd(f'sudo insmod {__DRIVER_LOC} || true', print_stdout=False, assert_retcode=True)
        cmd(f'sudo chmod 0666 /dev/sakura_*', assert_retcode=True)
        # Start dma daemon in the background
        pid = subprocess.Popen(['sudo', find_bin_util("ec_dma_daemon_proc")], close_fds=True).pid
        print(f'Started DMA daemon with PID {pid}')
        # Launch ddr init executable (non sudo)
        freq_arg = f' -f {int(args.sakura1_start)}' if args.sakura1_start != '' else ''
        cmd(find_bin_util("sakura_ddr_init") + freq_arg, print_stdout=True, assert_retcode=True)
        print(f'SUCCESS!')
        return 0
    arg_p.print_help()
    return 0

if __name__ == '__main__':
    sys.exit(main())
