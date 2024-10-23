import pandas as pd

categories = {
	"aime": 0, 
	"algebra": 0, 
	"amc": 0, 
	"induction": 0, 
	"mathd_algebra": 0, 
	"mathd_numbertheory": 0, 
	"numbertheory": 0,
	"imo": 0
}

df = pd.read_csv("results/execution.tsv", sep="\t")

cnt = 0
for i in range(len(df)):
	if df.loc[i]['Result'] == True:
		cnt += 1
		name = df.loc[i]['Name']
		cur_cat = "None"
		for cat in categories:
			if cat == name[:len(cat)]:
				cur_cat = cat
				break
		# print(cur_cat)
		categories[cur_cat] += 1

print(cnt)
print(categories)