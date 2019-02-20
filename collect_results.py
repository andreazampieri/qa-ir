import os
import json
import sys
import re 
import matplotlib.pyplot as plt

def pareto_frontier(a, b, name):
    l = sorted(list(zip(a,b,name)),reverse=True)
    p_front = [l[0]]    
    for point in l[1:]:
        if point[1] >= p_front[-1][1]:
            p_front.append(point)
    return p_front

if len(sys.argv) != 2:
	name = "testcnn"
else:
	name = sys.argv[1]

res = dict()
pattern = re.compile(name+"(?P<number>[0-9]+)")
for filename in os.listdir():
	match = pattern.match(filename)
	if match != None:
		number = int(match["number"])
		with open(filename+"/valid_metrics_best.json","r") as file:
			res[number] = json.load(file)

m_ap,m_rr, exp_name = [],[],[]
for k,v in res.items():
	m_ap.append(v["MAP"])
	m_rr.append(v["MRR"])
	exp_name.append(k)

best_exp = pareto_frontier(m_ap,m_rr,exp_name)
print("Best results in exp: ",str([v[-1] for v in best_exp]))
# plt.plot(m_ap,m_rr,"ro")
# plt.xlabel("MAP")
# plt.ylabel("MRR")
# plt.show()

for ap,rr, exp_no in best_exp:
	stats_file = f"{name}{exp_no}/"
	with open(stats_file,"r") as file:
		data = json.load(file)
		print(f"Experiment {exp_no}:\tmAP: {ap}\tmRR: {rr}")
		print(f"Model params : {data['model']['params']}")
		print(f"Optimizer: {data['optimizer']}")