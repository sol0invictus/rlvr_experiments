"""Compare ARS results between two JSON files."""
import json
import sys

pre_path = sys.argv[1]
post_path = sys.argv[2]

with open(pre_path) as f:
    pre = json.load(f)
with open(post_path) as f:
    post = json.load(f)

print(f"{'ATOM':28s}| {'PRE':>6s}| {'POST':>6s}| {'DELTA':>7s}")
print("-" * 55)
for atom in pre['per_atom']:
    pre_acc = pre['per_atom'][atom]['accuracy'] * 100
    post_acc = post['per_atom'][atom]['accuracy'] * 100
    delta = post_acc - pre_acc
    marker = " ***" if abs(delta) > 5 else ""
    print(f"{atom:28s}| {pre_acc:5.1f}%| {post_acc:5.1f}%| {delta:+5.1f}%{marker}")
print("-" * 55)
print(f"{'L1 avg':28s}| {pre['L1_avg']*100:5.1f}%| {post['L1_avg']*100:5.1f}%| {(post['L1_avg']-pre['L1_avg'])*100:+5.1f}%")
print(f"{'L2 avg':28s}| {pre['L2_avg']*100:5.1f}%| {post['L2_avg']*100:5.1f}%| {(post['L2_avg']-pre['L2_avg'])*100:+5.1f}%")
print(f"{'ARS':28s}| {pre['ARS']*100:5.1f}%| {post['ARS']*100:5.1f}%| {(post['ARS']-pre['ARS'])*100:+5.1f}%")
