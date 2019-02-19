import os
import json
import sb
import re 
import pyplot as plt

def pareto_frontier(a, b, name):
    l = sorted(list(zip(a,b,name)))
    p_front = [l[0]]    
    for point in l[1:]:
        if point[1] >= p_front[-1][1]:
            p_front.append(point)
    return p_front

if len(sb.argv) != 2:
	name = "testcnn"
else:
	name = sb.argv[1]

res = dict()
pattern = re.compile(name+"(?P<number>[0-9]+)")
for filename in os.listdir():
	match = pattern.match(filename)
	if match != None:
		number = int(match("number"))
		with open(filename+"/valid_metrics_best.json","r") as file:
			res[number] = json.load(file)

m_ap,m_rr, exp_name = [],[],[]
for k,v in res.items():
	m_ap.append(v["MAP"])
	m_rr.append(v["MRR"])
	exp_name.append(k)

best_exp = pareto_frontier(m_ap,m_rr,exp_name)
print("Best results in exp: ",str([v[-1] for v in best_exp]))
plt.plot(list(m_ap.values()),list(m_rr.values()))
plt.xlabel("MAP")
plt.ylabel("MRR")
plt.show()