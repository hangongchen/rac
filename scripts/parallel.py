# helper function to parallelize render jobs over gpus
# python scripts/parallel.py "0" cat76 cat76 scripts/extract_and_render_mesh.sh
import sys
import configparser
import pdb
import os
import shutil
import subprocess
import shlex

devs=sys.argv[1]
seqname=sys.argv[2]
loadname=sys.argv[3]
scriptpath=sys.argv[4]

devs=devs.split(",")
num_dev = len(devs)
use_screen = shutil.which("screen") is not None

config = configparser.RawConfigParser()
config.read('configs/%s.config'%seqname)

model_path = 'logdir/%s/params_latest.pth'%loadname

vid_groups = {}
for vidid in range(len(config.sections())-1):
    dev = devs[vidid%num_dev]
    if dev in vid_groups.keys():
        vid_groups[dev] += ' %s'%vidid
    else:
        vid_groups[dev] = '%s'%vidid

procs = {}
for dev in devs:
    cmd_args = [
        "bash",
        "scripts/sequential_exec.sh",
        dev,
        seqname,
        model_path,
        vid_groups[dev],
        scriptpath,
    ]

    if use_screen:
        cmd = 'screen -dmS "render-%s-%s" bash -c "%s"' % (
            seqname,
            dev,
            " ".join(shlex.quote(arg) for arg in cmd_args),
        )
        print(cmd)
        err = os.system(cmd)
        if err:
            print("FATAL: command failed")
            sys.exit(err)
    else:
        print("screen not found; running without detaching:", " ".join(shlex.quote(arg) for arg in cmd_args))
        procs[dev] = subprocess.Popen(cmd_args)

if not use_screen:
    failed = False
    for dev, proc in procs.items():
        ret = proc.wait()
        if ret != 0:
            print("FATAL: command failed for dev %s with exit code %s" % (dev, ret))
            failed = True
    if failed:
        sys.exit(1)
