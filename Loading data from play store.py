import torch

import json
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from pygments import highlight
from pygments.lexers import JsonLexer

from pygments.formatters import TerminalFormatter

from google_play_scraper import Sort, reviews, app




%config InlineBackend.figure_format='retina'

sns.set (style='whitegrid',palette='muted',font_scale=1.2)

app_packages=['com.anydo','com.todoist','com.ticktick.task','com.habitrpg.android.habitica','cc.forestapp','com.oristats.habitbull',
              'com.levor.liferpgtasks','com.habitnow','com.microsoft.todos','prox.lab.calclock','com.gmail.jmartindev.timetune','com.artfulagenda.app',
              'com.tasks.android','com.appgenix.bizcal','com.appxy.planner']

len(app_packages)

app_info=[]

for i in tqdm(app_packages):
    info=app(i,lang="en",country="us")
    del info['comments']
    app_info.append(info)


def print_json(json_object):
    json_str= json.dumps(
    json_object,
    indent=2,
    sort_keys=True,
    default=str)
    print(highlight(json_str, JsonLexer(), TerminalFormatter()))

print_json(app_info[0])


fig,axs=plt.subplots(2,(len(app_info)//2),figsize=(10,3))

for i,j in enumerate(axs.flat):
    ai=app_infos[i]
    img=plt.imread(ai['icon'])
    j.imshow(img)
    j.set_title(ai['title'][:10])
    j.axis('off')


app_info_df=pd.DataFrame(app_info)

app_info_df.head()

app_info_df.to_csv("app.csv",index=None,header=True)

app_reviews=[]
for ap in tqdm(app_packages):
    for score in range(1,6):
        for sort_order in [Sort.MOST_RELEVANT,Sort.NEWEST]:
            rvs,_= reviews(ap,
                       lang="en",
                        country="us",
                       sort=sort_order,
                       count=200 if score==3 else 100,
                       filter_score_with=score)
            for r in rvs:
                r['sortOrder']='most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
                r['appId']= ap
            app_reviews.extend(rvs)

print_json(app_reviews[0])

app_reviews_df=pd.DataFrame(app_reviews)

app_reviews_df.shape


app_reviews_df.head()

app_reviews_df.to_csv("review.csv",index=None,header=True)
