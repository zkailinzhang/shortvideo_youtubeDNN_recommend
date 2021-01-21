

class SuperPrams:
    #### config.py
    user_count = 100000
    item_count = 50000
    category_count = 3000
    tags_count = 100

    item_his_max_len =15
    lr =0.0001
    epoches = 5
    batch_size = 128

    dnn_layers =3
    dnn_nums =[200,40,64]
    nums_sampled=100 #??


    event_name = 'click'
    start_date =''
    end_date = ''
    start_time = ''
    end_time = ''

    emb_dims = 64
    model_save_path ='model'
    items_matrix_save_path = 'matrix' 