import argparse, json, pathlib, platform, subprocess
p=argparse.ArgumentParser();p.add_argument('--metal',action='store_true');args=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];out=root/'build';out.mkdir(exist_ok=True)
common=['clang','-O3','-fPIC']
cmd=common+(['-dynamiclib'] if platform.system()=='Darwin' else ['-shared'])+[str(root/'native/cpu.c'),'-o',str(out/'cpu.so')]
subprocess.run(cmd,check=True)
commands=[cmd]
if args.metal:
    if platform.system()!='Darwin':raise SystemExit('Metal requires macOS')
    cmd=common+['-dynamiclib','-fobjc-arc',str(root/'native/metal.m'),'-framework','Foundation','-framework','Metal','-o',str(out/'metal.so')]
    subprocess.run(cmd,check=True);commands.append(cmd)
(out/'build.json').write_text(json.dumps({'commands':[[arg.replace(str(root)+'/','') for arg in cmd] for cmd in commands],'compiler':subprocess.check_output(['clang','--version'],text=True)},indent=2))
print('Built',out)
