# YoutubeMatch

youtubednn 召回深度模型

## 环境
tensorflow-gpu==1.15
faiss-gpu==1.6.3

## 完成
目前代码是一个完整数据流的代码,
-从events和items表拉数据,手动保存到csv,
- 预处理 空置填充,,特征(category,tags等)  构建索引文件,
- 样本构造:训练样本,测试样本; 
- 生成最终的物品矩阵,保存到weight.npy,用于线上服务,
- 把weight文件,喂入Faiss构建索引,然后用户特征喂入输入用户向量,快速取TopK,
- 计算testdata的 racall,hit,可分别计算top5,top10,top20,top30,来验证模型的效果,达到预设阈值稳定后上线
- docker tfserving 完成
  

## tfserving 
- match.py 添加tfserving代码
- online.py 在线请求代码
- docker tf/serving ,完成
    
    - 起服务
    ```bash
    docker run -t --rm -p 8051:8051 -v "/home/tv/code/tv/youtube/serving:/models/youtube23" -e MODEL_NAME=youtube23 tensorflow/serving &

    ```
    - 请求
    ```
    curl -d '{"instances": [{"user_id_ph":40252,"his_item_seq_ph":[1971,363,565],"his_cate_seq_ph":[807,657,657],
    "his_tag_seq_ph":[1579,657,940],"his_len_ph": 3},{"user_id_ph":40252,"his_item_seq_ph":[1971,363,565],"his_cate_seq_ph":[807,657,657],
    "his_tag_seq_ph":[1579,657,940],"his_len_ph": 3}]}'     -X POST http://localhost:8501/v1/models/youtube23:predict
    ```
    返回:
    ```
    {
    "predictions": [[8.79667423e-05, -0.00212874915, -0.000757557689, -0.00103763444, 0.0131146498, 0.00752771692, 0.0152459918, -0.00107731682, 0.0162343159, 0.00259575015, -0.00177589757, -0.000561485067, 0.00530720362, 0.0101294555, -0.00281065307, -0.00203643111, 0.00220618444, 0.00265264581, 0.00319370744, 0.000664713618, -0.00252806908, -0.00166927197, 0.0026902596, -0.00314125954, 0.00262762117, -0.00015888935, -0.000598910847, 0.0162613597, -0.000365366781, -0.00108401617, 0.00761482073, 0.00626033125, 5.73499929e-05, 0.0136919543, 0.00057245692, -0.000511030725, -0.000258882414, -6.19707644e-05, 0.000438784016, 0.000344205357, 0.00966643263, 0.00682006916, -0.00217664917, -0.000838903, 0.00419464149, 0.00172620697, -0.0023352732, -0.00188220211, -0.000906770932, -0.00156356, 0.000387862092, -0.00232434692, 0.00628324, 0.0233979933, -0.00136896502, 0.000955678872, 0.00696203858, -0.002295641, -0.00253539719, -0.00103426038, -0.000906979782, -0.00223148381, 0.00464658, 0.00440197345], [8.79667423e-05, -0.00212874915, -0.000757557689, -0.00103763444, 0.0131146498, 0.00752771692, 0.0152459918, -0.00107731682, 0.0162343159, 0.00259575015, -0.00177589757, -0.000561485067, 0.00530720362, 0.0101294555, -0.00281065307, -0.00203643111, 0.00220618444, 0.00265264581, 0.00319370744, 0.000664713618, -0.00252806908, -0.00166927197, 0.0026902596, -0.00314125954, 0.00262762117, -0.00015888935, -0.000598910847, 0.0162613597, -0.000365366781, -0.00108401617, 0.00761482073, 0.00626033125, 5.73499929e-05, 0.0136919543, 0.00057245692, -0.000511030725, -0.000258882414, -6.19707644e-05, 0.000438784016, 0.000344205357, 0.00966643263, 0.00682006916, -0.00217664917, -0.000838903, 0.00419464149, 0.00172620697, -0.0023352732, -0.00188220211, -0.000906770932, -0.00156356, 0.000387862092, -0.00232434692, 0.00628324, 0.0233979933, -0.00136896502, 0.000955678872, 0.00696203858, -0.002295641, -0.00253539719, -0.00103426038, -0.000906979782, -0.00223148381, 0.00464658, 0.00440197345]
    ]
    }
    ```


## todo
- 数据量少
- 现在tags单值,加上多值
- 更多特征,desc..

