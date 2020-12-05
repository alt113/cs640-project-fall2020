import os
import pandas as pd


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def find_img_index(iname, emotion_list):
    return emotion_list.index(iname)


PATH_to_DIR = os.getcwd()

df = pd.read_csv(PATH_to_DIR + '/data/legend.csv')

# del df['user.id']
#
# emotions = dict.fromkeys(df['emotion'])
#
# print('* unique emotions (unprocessed) :')
# for k in emotions.keys():
#     print(f"\t* {k}")
#
# df['emotion'] = df['emotion'].apply(lambda x: str(x).lower())
#
# emotions = dict.fromkeys(df['emotion'])
# print('* unique emotions (processed) :')
# for k in emotions.keys():
#     print(f"\t* {k}")
#
# for idx, em in enumerate(df['emotion']):
#     if em in ['anger', 'disgust', 'fear', 'sadness', 'contempt']:
#         df.iloc[idx, 1] = 'negative'
#     elif em in ['surprise', 'happiness']:
#         df.iloc[idx, 1] = 'positive'
#
# emotions = dict.fromkeys(df['emotion'])
# print('* unique emotions (after mapping) :')
# for k in emotions.keys():
#     print(f"\t* {k}")
#
# print(df.head(10))
#
# df.to_csv(PATH_to_DIR + '/data/legend.csv')

del df['Unnamed: 0']
print(df.head())

list_of_image_names = os.listdir(PATH_to_DIR + '/data/images/')

emotions = df['emotion']

print(f"emotion data size: {len(emotions)}\nimage data size: {len(list_of_image_names)}")

print(f"Length of intersection: {len(intersection(list_of_image_names, df['image']))}")

# files_to_delete = []
# intersect = intersection(list_of_image_names, df['image'])
#
# for f in list_of_image_names:
#     if f not in intersect:
#         files_to_delete.append(f)
#
# print(f"First 10 files to delete: {files_to_delete[0:10]}")
#
# for iname in files_to_delete:
#     os.remove(PATH_to_DIR + '/data/images/' + iname)
#
# print('** deleted un-labeled files **')

for img_name in list_of_image_names:
    idx = find_img_index(img_name, list(df['image']))

    emotion_label = df.iloc[idx, 1]
    raw_name = img_name.split('.jpg')[0]

    new_name = raw_name + '_' + emotion_label + '.jpg'

    os.rename(PATH_to_DIR + '/data/images/' + img_name,
              PATH_to_DIR + '/data/images/' + new_name)

print('** double check with first few name **')
print(os.listdir(PATH_to_DIR + '/data/images/')[0:10])
