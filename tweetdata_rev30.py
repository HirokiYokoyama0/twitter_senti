#　 ver24 清書プログラム
#  フォロワー数,RT,いいね数も取ってくる
#  フォロワー数がすくないユーザも除外する処理を追加
#  ファイルの複数回処理（csvdata処理前にあるファイル）
#  220706 rankingに順位のデータ列を付加 
#  220707 rankingを100位までに変更　
#  220707 前提条件に清書用名詞の総数を表示を追加、清書分かち書きcsvフォルダ作成を追加　
#  220708 形態素解析結果を、一行ずつ清書csvについて ## 追加したものの処理がおそいため、もとのゆきこのやり方を採用　変更点#$で検索
#  rev27 220709 各政策統合リストに載っている統合カウント処理を追加  
#  rev28 220709 ↑の割合を追加
#  rev30 220710ツイート本文を一行ずつ感情分析して、清書csvに付加

from re import U
import re
from urllib.parse import uses_relative
import warnings
import os
import glob #globモジュールを宣言

### 感情分析用のモジュール
from transformers import pipeline 
from transformers import AutoModelForSequenceClassification 
from transformers import BertJapaneseTokenizer

warnings.filterwarnings('ignore')

from janome.tokenizer import Tokenizer
import csv
import itertools
from collections import Counter
from itertools import chain
import pandas as pd

# 初期設定
removeword = ['参院選','参院','選挙','こと','さん','これ','よう','まま','それ','ところ','ため','そう','ほか','みたい','たち','もの']

tweet_csvfiles = glob.glob("csvdata処理前/tweetdata_*.csv")

# print("aaa",tweet_csvfiles)
# csvdata処理後フォルダが存在しない場合に作成

if os.path.isdir("csvdata処理後") == False :
    os.mkdir("csvdata処理後")

if os.path.isdir("csvdata除去データ") == False :
    os.mkdir("csvdata除去データ")

if os.path.isdir("csvdata清書データ") == False :
    os.mkdir("csvdata清書データ")

if os.path.isdir("前提条件") == False :
    os.mkdir("前提条件")

# meishi関数の定義
def meishi(text):

    #t = Tokenizer()
    
    t = Tokenizer("user_dic02.csv",udic_enc="utf8")
    tokens = t.tokenize(text)
    noun = []
    for token in tokens:
        partOfSpeech = token.part_of_speech.split(",")[0]
        if partOfSpeech == "名詞":
            noun.append(token.surface)
    return noun

def sentiment(text):
    #TARGET_TEXT = "自民党に投票するつもり"
 
    model = AutoModelForSequenceClassification.from_pretrained('daigo/bert-base-japanese-sentiment') 
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking') 
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer) 

    senti_anal = nlp(text)
    #print(nlp(text))

    return senti_anal
 
    
    

### 複数回処理開始 ###
for csvfile in tweet_csvfiles:
# データ(ユーザー名、本文の列のみ)を読み込み
    print('⭐️処理を開始⭐️',csvfile)
    csvdata = pd.read_csv(csvfile, usecols=['user.screen_name', '本文','ユーザフォロワー数','いいね','リツイート数'], encoding='cp932')
    csvdata = csvdata.reindex(columns=['user.screen_name', 'ユーザフォロワー数','いいね','リツイート数','本文'])


     # outputファイルの部分的な名前を決める
     # tweetdata_2022-06-30_230_参院(300)_分析用.csv →　”2022-06-30_230_参院(300)_分析用” だけ取り出す
    outputfilename = csvfile.split('\\') #ファイル名を切り出し  ★mac('/')・win('\\')変更　これは備忘録だから消さないで
    outputfilename = outputfilename[1].split('.') #ファイルの途中の名前切り出し
    outputfilename = outputfilename[0][10:] #ファイルの途中の名前切り出し


    # 読み込んだデータ数を表示
    print('⭐️データ総数を表示⭐️========', len(csvdata))

    # 読み込んだデータを表示
    #print(csvdata)
    print("最初の5行_1\n",csvdata.head())

    # ユーザー名の出現数を調べる
    user_count = csvdata['user.screen_name'].value_counts()
    #print(user_count)
    #print(type(user_count))

    # カラム名を変更
    counttable = user_count.reset_index(name='count')
    print("出現数ランキング最初の5行_2\n",counttable.head())

    # 除去対象ユーザー(4以上ツイート)を抽出
    exclusion_user = counttable[counttable['count'] > 3]
    exclusion_user.to_csv('exclusion_user.csv', encoding='cp932') #除外ユーザ候補をcsvデータに吐き出し
    #print(exclusion_user)
    print('⭐️除去人数を表示１⭐️========', len(exclusion_user))

    # pandas型の除去対象ユーザーリストをpython標準のリスト型へ変換
    remove_Userlist = exclusion_user['index'].values.tolist()
    
    #除去データ数を計算
    removed_data = pd.DataFrame(exclusion_user, columns=['count'])
    sum_column = removed_data.sum(axis = 0)
    print('⭐️除去ツイート数を表示⭐️========', sum_column)

    # csvdataから除去対象ユーザーを削除して csvdata_revise0 = 清書データとする
    csvdata_revise0 = csvdata[~csvdata["user.screen_name"].isin(remove_Userlist)]
    # 清書データ数を表示
    print('⭐️除外後のデータ総数を表示(rev0)⭐️========', len(csvdata_revise0))

    # 前提条件ファイルの名前を決める
    filename_cond = '前提条件/prerequisites_'+ outputfilename + '.txt'
    
    # 清書データ一覧を表示
    #print("csvdata_revise0 →\n",csvdata_revise0)

    # 除去対象ユーザーのcsvデータを作成
    csvdata_removed = csvdata[csvdata["user.screen_name"].isin(remove_Userlist)]
    csvdata_removed.to_csv('csvdata除去データ/csvdata_removed' + outputfilename + '.csv', encoding='cp932') #csvdata_removedをcsvデータに吐き出し


     # ツイート本文を統合:$$$ここを修正していく、感情分析を付加する
    integrate = ""
    meishiwd2 = []
    senti_res_label = []
    senti_res_score = []

    for index, row in csvdata_revise0.iterrows():

        print("--->本文",row["本文"])
        print("--->感情分析結果",sentiment(row["本文"]))

        senti_res = sentiment(row["本文"])
        print("type,",type(senti_res))
        senti_res = senti_res[0]
        
        senti_res_label.append(senti_res["label"])
        senti_res_score.append(senti_res["score"])

        meishiwd2.append(meishi(row["本文"]))  #$ この一行ずつの処理はめちゃんこおそい。ゆきこがやっていたような結合してからの処理に変更したほうがいいかも。
        #print("--->kanrenwd ",meishiwd2)
        
        integrate += row["本文"]

    csvdata_revise0["ポジネガ"] = senti_res_label
    csvdata_revise0["スコア"] = senti_res_score
    csvdata_revise0["関連ワード"] = meishiwd2 #$ 

    print("---> csvdata_revise0 ",csvdata_revise0.head)
    # 清書データをcsvに書き出し
    csvdata_revise0.to_csv('csvdata清書データ/csvdata_revise0' + outputfilename + '.csv', encoding='cp932') #csvdata_revise0をcsvデータに吐き出し

    # csvdata処理後フォルダに入れるファイル名を決める
    filename_tw = 'csvdata処理後/integtxt'+ outputfilename + '.txt'
    filename_rank = 'csvdata処理後/ranking' + outputfilename + '.csv'
    filename_wakachi = 'csvdata処理後/wakachi'+ outputfilename + '.csv'
    filename_rankGenru = 'csvdata処理後/rankGenru'+ outputfilename + '.csv'

    # 統合したツイート本文のデータをtxtファイルに書き出し
    with open(filename_tw, 'w', encoding='utf-8', newline='\n', errors='ignore') as f:
        f.writelines(integrate)

    # テキスト処理
    kanren_word = [] # 空のリスト作成

    text = integrate 
    meishiwd = meishi(text)
    # meishiwdに注目ワードが含まれていたら、”ツイート本文”
    kanren_word.append(meishiwd)

    kanren_word2 = list(itertools.chain.from_iterable(kanren_word))
    #$kanren_word2 = list(itertools.chain.from_iterable(meishiwd2)) ##処理がおそいからやめた・・・↑のゆきこ統合処理を採用
    kanren_word3=[]

    for s in kanren_word2:
        if s not in removeword and len(s) > 1 : 
            kanren_word3.append(s)

    print('⭐️清書データ処理後名詞総数を表示⭐️========', len(kanren_word3))


    ##ワード集約処理 例　自民→自民党に置換　
    kanren_word4 = [re.sub("^自民$", "自民党",s) for s in kanren_word3]
    kanren_word4 = [re.sub("^自由民主党$", "自民党",s) for s in kanren_word4]
    kanren_word4 = [re.sub("^立憲$", "立憲民主",s) for s in kanren_word4]
    kanren_word4 = [re.sub("^立民$", "立憲民主",s) for s in kanren_word4]
    kanren_word4 = [re.sub("^公明$", "公明党",s) for s in kanren_word4]
    kanren_word4 = [re.sub("^共産$", "共産党",s) for s in kanren_word4]
    kanren_word4 = [re.sub("^社民$", "社民党",s) for s in kanren_word4]
    kanren_word4 = [re.sub("^N党$", "NHK党",s) for s in kanren_word4]
    kanren_word4 = [re.sub("^N党$", "NHK党",s) for s in kanren_word4]
    kanren_word4 = [re.sub("^N国$", "NHK党",s) for s in kanren_word4]
    kanren_word4 = [re.sub("^N国$", "NHK党",s) for s in kanren_word4]
   

    Count_kanrenword = Counter(kanren_word4)
    ranking = Count_kanrenword.most_common(100)
    
    #print("Count_kanrenword->",Count_kanrenword)
    #print("Count_kanrenwords_kazu---->",Count_kanrenword["自民党"])
    #print("c5---->",Count_kanrenword["aaaaa"])
    #print("置換後->",ranking)

    
    # 名詞ランキングをcsvファイルに書き出し
    with open(filename_rank, 'w', newline='',encoding='CP932',errors='ignore') as f2:

        f2.write("共起ワード"+ ','+ "出現回数"+ ','+ "順位" '\n') #一行目にindexをつける
        for i,word in enumerate(ranking):
            f2.write(word[0]+ ','+ str(word[1]) + ',' + str(i+1) + '\n')

    ##### 各政策統合リストキーワード分類の実施　→　例：”消費税”、”憲法”　→　”施策動向”　に集約

    aggri_words = {"憲法関連": ["憲法","改憲","護憲"],"経済消費": ["経済","消費税","物価高","増税","円安"],"エネルギー": ["エネルギー","節電","電力","原発","再稼働"],"社会保障": ["社会保障","年金","社会保険","失業"],"外交安全": ["外交","安全保障","安保","国防"],"新型コロナ": ["新型コロナ","コロナ","感染症","ワクチン"],"教育": ["教育","子育て","教育無償","育児"]}

    #print(aggri_words["憲法関連"][1])
    with open(filename_rankGenru, 'w', newline='',encoding='CP932',errors='ignore') as frg:
        frg.write("合計/小計" + "," + "政策ジャンル"+ ','+ "出現回数" + ',' + "出現割合" + '\n') #一行目にindexをつける
        for aggri_word in aggri_words.keys():   #aggri_wordnに辞書のkeyが入る　"憲法関連","経済消費",社会保障"
            #print("--区切り--")
            wordsum=0
            for element in aggri_words[aggri_word]: # aggri_words からすべて要素["憲法","改憲","護憲"]を引っ張ってくる
            #print(element)
                elementwariai = Count_kanrenword[element]/len(kanren_word3)*100
                #print("集約ワード(要素)ごと出現回数",element,Count_kanrenword[element],elementwariai, sep=':')
                
                frg.write('小計'+ ','+ element + ',' + str(Count_kanrenword[element]) + ',' + str(elementwariai) + '\n')
                wordsum += Count_kanrenword[element]
            #print(aggri_word,"合計",wordsum,sep='::')
            wordsumwariai = wordsum/len(kanren_word3)*100
            frg.write('合計'+ ','+ aggri_word + ',' + str(wordsum)  + ',' + str(wordsumwariai)+ '\n')


     # 前提条件ファイルを作り、必要な条件をテキストに書き出し
    with open(filename_cond, 'w', encoding='utf-8', newline='\n', errors='ignore') as fc:
        fc.write('データ総数を表示 => ' +  str(len(csvdata)) + '\n')
        fc.write('除去人数を表示 => ' +  str(len(exclusion_user)) + '\n')
        #fc.write('除去tweet数を表示 => ' +  str(len(sum_column)) + '\n')
        fc.write('除外後のデータ総数を表示 => ' +  str(len(csvdata_revise0)) + '\n')
        fc.write('清書データ処理後の名詞総数を表示 => ' +  str(len(kanren_word3)) + '\n')


    # 分かち書きデータをcsvファイルに書き出し(ダメだった)
    with open(filename_wakachi, 'w', newline='',encoding='CP932',errors='ignore') as f3:
        for d in kanren_word3 :
            f3.write("%s\n" % d)






