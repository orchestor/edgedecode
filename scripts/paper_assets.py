"""Regenerate vector figures and LaTeX tables from committed evidence."""
from pathlib import Path
import json,math
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'paper';D=ROOT/'results/pilot';F=P/'figures';F.mkdir(exist_ok=True)
ops=json.loads((D/'operators.json').read_text());search=json.loads((D/'search.json').read_text());arch=json.loads((D/'architecture.json').read_text())
stage=json.loads((ROOT/'results/m4-stage2/summary.json').read_text())
navy=HexColor('#12334d');teal=HexColor('#167d8d');orange=HexColor('#bd6527');gray=HexColor('#62717c')
def setup(name,title):
 c=canvas.Canvas(str(F/name),pagesize=(480,252));c.setFillColor(navy);c.setFont('Helvetica-Bold',12);c.drawString(44,230,title);return c
def axes(c,x0=55,y0=45,x1=455,y1=200):
 c.setStrokeColor(gray);c.setLineWidth(.5);c.line(x0,y0,x1,y0);c.line(x0,y0,x0,y1)

rows=[r for r in ops['rows'] if r['workload']=='m1024_k1024_n1' and r['backend']=='cpu']
rows.sort(key=lambda r: (r['bits']!=32,r['bits'],r['group']))
c=setup('operator.pdf','CPU reference operator: M=K=1024, N=1');axes(c)
maximum=max(r['median_ms'] for r in rows)*1.25
for j in range(5):
 val=maximum*j/4;y=45+155*j/4;c.setFont('Helvetica',8);c.setFillColor(gray);c.drawRightString(50,y-3,f'{val:.2f}');c.setStrokeColor(HexColor('#e4e9ed'));c.line(55,y,455,y)
for i,r in enumerate(rows):
 x=67+i*55;height=r['median_ms']/maximum*155;c.setFillColor(teal if r['bits']==4 else navy);c.rect(x,45,33,height,fill=1,stroke=0)
 low=45+r['p25_ms']/maximum*155;high=45+r['p75_ms']/maximum*155;c.setStrokeColor(orange);c.setLineWidth(1);c.line(x+16,low,x+16,high);c.line(x+12,low,x+20,low);c.line(x+12,high,x+20,high)
 c.setFillColor(gray);c.setFont('Helvetica',7);c.drawCentredString(x+16,32,r['format'].replace('_','/'))
c.setFont('Helvetica',8);c.drawString(55,213,'Milliseconds; median with interquartile range, nine hot-input repetitions');c.drawString(55,13,'Unoptimized single-thread implementation; not a peak-hardware comparison.');c.save()

c=setup('search.pdf','Synthetic network: independent test error versus CPU time');axes(c)
sel=search['rows'];xmax=max(r['median_ms'] for r in sel)*1.1;xmin=min(r['median_ms'] for r in sel)*.85
lo,hi=-4,-.5
for v in [-4,-3,-2,-1]:
 y=45+(v-lo)/(hi-lo)*155;c.setFillColor(gray);c.setFont('Helvetica',8);c.drawRightString(50,y-3,'1e'+str(v));c.setStrokeColor(HexColor('#e4e9ed'));c.line(55,y,455,y)
for i in range(5):
 x=55+i*100;v=xmin+(xmax-xmin)*i/4;c.setFillColor(gray);c.drawCentredString(x,32,f'{v:.3f}')
colors={'hardware_aware':orange,'quality_only':navy,'uniform_q4_g128':HexColor('#b24646'),'uniform_q8_g64':teal,'random_feasible':gray}
for r in sel:
 x=55+(r['median_ms']-xmin)/(xmax-xmin)*400;y=45+(math.log10(r['test_relative_mse'])-lo)/(hi-lo)*155
 c.setFillColor(colors[r['policy']]);c.circle(x,y,3.2,fill=1,stroke=0)
for i,(name,color) in enumerate(colors.items()):
 x=46+(i%3)*145;y=215-(i//3)*12;c.setFillColor(color);c.circle(x,y,2.5,fill=1,stroke=0);c.setFont('Helvetica',7);c.drawString(x+6,y-2,name.replace('_',' '))
c.setFillColor(gray);c.setFont('Helvetica',8);c.drawString(55,13,'x: composed CPU milliseconds; y: relative MSE (log scale), three seeds.');c.save()

c=setup('architecture.pdf','Hypothetical scenarios: throughput feature benefit');axes(c)
items=[r for r in arch['rows'] if r['bits']==4 and r['group']==32 and r['feature']=='double_decode']
maxspeed=max(r['speedup'] for r in items)*1.1
for i in range(5):
 y=45+155*i/4;c.setFillColor(gray);c.setFont('Helvetica',8);c.drawRightString(50,y-3,f'{maxspeed*i/4:.1f}');c.setStrokeColor(HexColor('#e4e9ed'));c.line(55,y,455,y)
for i,r in enumerate(items):
 x=66+i*43;c.setFillColor(teal);c.rect(x,45,22,r['speedup']/maxspeed*155,fill=1,stroke=0);c.setFont('Helvetica',6);c.setFillColor(gray);c.drawCentredString(x+11,32,r['profile'].split('_')[0]);c.drawCentredString(x+11,23,'N='+str(r['n']))
c.setFont('Helvetica',8);c.drawString(55,213,'Speedup from doubling decode throughput; no calibration to any device');c.drawString(55,9,'All profiles and bottlenecks are assumptions; no energy/area gain established.');c.save()

def tex(s):return str(s).replace('_',r'\_')
lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Format & KiB & Median ms & IQR ms & Rel. MSE \\',r'\midrule']
for r in rows:lines.append(f"{tex(r['format'])} & {r['weight_bytes']/1024:.0f} & {r['median_ms']:.3f} & {r['p75_ms']-r['p25_ms']:.3f} & {r['relative_mse']:.2e} \\")
# Use explicit LaTeX row separators (avoid shell/string ambiguity).
lines=[l+'\\' if l.endswith(' \\') and not l.endswith('\\\\') else l for l in lines]
lines += [r'\bottomrule',r'\end{tabular}'];(P/'operator-table.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{tabular}{rlrrr}',r'\toprule',r'Seed & Selection & Dev. MSE & Test MSE & CPU ms \\',r'\midrule']
for r in sel:
 if r['policy'] in ('hardware_aware','uniform_q4_g128','uniform_q8_g64'):
  name={'hardware_aware':'HW-aware','uniform_q4_g128':'Q4/G128','uniform_q8_g64':'Q8/G64'}[r['policy']]
  lines.append(f"{r['seed']} & {name} & {r['dev_relative_mse']:.5f} & {r['test_relative_mse']:.5f} & {r['median_ms']:.3f} "+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(P/'search-table.tex').write_text('\n'.join(lines)+'\n')
(P/'counts.tex').write_text(f"\\newcommand{{\\OpCount}}{{{len(ops['rows'])}}}\n\\newcommand{{\\SkipCount}}{{{len(ops['skipped'])}}}\n\\newcommand{{\\ScenarioCount}}{{{len(arch['rows'])}}}\n")

# Stage 2: real-model throughput, split by execution phase and backend.
c=setup('stage2-performance.pdf','Qwen2.5-1.5B: measured throughput by phase')
axes(c,55,54,455,196)
groups=[('CPU prefill',stage['performance']['cpu']['pp2048']),
        ('CPU decode',stage['performance']['cpu']['tg128']),
        ('Metal prefill',stage['performance']['metal']['pp2048']),
        ('Metal decode',stage['performance']['metal']['tg128'])]
maxv=max(x['tokens_per_second'] for _,g in groups for x in g.values())*1.12
for j in range(5):
 val=maxv*j/4;y=54+142*j/4;c.setFillColor(gray);c.setFont('Helvetica',7);c.drawRightString(50,y-3,f'{val:.0f}');c.setStrokeColor(HexColor('#e4e9ed'));c.line(55,y,455,y)
palette={'F16':navy,'Q8_0':teal,'Q4_K_M':orange}
for gi,(label,g) in enumerate(groups):
 base=70+gi*98
 for mi,model in enumerate(('F16','Q8_0','Q4_K_M')):
  value=g[model]['tokens_per_second'];x=base+mi*20
  c.setFillColor(palette[model]);c.rect(x,54,15,value/maxv*142,fill=1,stroke=0)
 c.setFillColor(gray);c.setFont('Helvetica',6.5);c.drawCentredString(base+20,42,label)
for i,(model,color) in enumerate(palette.items()):
 x=142+i*78;c.setFillColor(color);c.rect(x,212,8,8,fill=1,stroke=0);c.setFillColor(gray);c.setFont('Helvetica',7);c.drawString(x+12,213,model.replace('_',r'\_'))
c.setFillColor(gray);c.setFont('Helvetica',7);c.drawString(55,19,'tokens/s; seven llama.cpp repetitions. Prefill uses 2048 tokens; decode uses 128.')
c.save()

lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Backend & Phase & F16 & Q8\_0 & Q4\_K\_M \\',r'\midrule']
for backend in ('cpu','metal'):
 for test,label in (('pp2048','Prefill'),('tg128','Decode')):
  g=stage['performance'][backend][test]
  lines.append(f"{backend.upper()} & {label} & {g['F16']['tokens_per_second']:.1f} & {g['Q8_0']['tokens_per_second']:.1f} & {g['Q4_K_M']['tokens_per_second']:.1f} "+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(P/'stage2-performance-table.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Format & Size (GiB) & WikiText PPL & HellaSwag (\%) \\',r'\midrule']
for model in ('F16','Q8_0','Q4_K_M'):
 m=stage['models'][model];lines.append(f"{tex(model)} & {m['file_bytes']/2**30:.2f} & {m['perplexity']:.4f} & {m['hellaswag_acc_norm_percent']:.1f} "+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(P/'stage2-quality-table.tex').write_text('\n'.join(lines)+'\n')
print('Generated 4 vector figures, 4 data tables and counts.')
