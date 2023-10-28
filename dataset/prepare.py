import os
import datasets

class DataPreparing:
    def __init__(self, save_data_dir, source_lang, target_lang):
        self.save_data_dir = save_data_dir
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    def download_dataset(self):
        # print(os.path.join(os.getcwd(), self.save_data_dir))
        # print(os.path.exists(os.path.join(os.getcwd(), self.save_data_dir)))
        assert os.path.exists(self.save_data_dir) == True
        if len(os.listdir(self.save_data_dir)) ==0:
            print('#1-Download Dataset')
            corpus = datasets.load_dataset("mt_eng_vietnamese", "iwslt2015-en-vi")
            
            print('#2-Save Dataset')
            for data in ['train', 'validation', 'test']:

                source_data, target_data = self.get_data(corpus[data])

                print('Source lang: {} - {}: {}'.format(self.source_lang, data, len(source_data)))
                print('Target lang: {} - {}: {}'.format(self.target_lang, data, len(target_data)))

                self.save_data(source_data, os.path.join(self.save_data_dir, data + '.' + self.source_lang))
                self.save_data(target_data, os.path.join(self.save_data_dir, data + '.' + self.target_lang))

        else:
            print('Dataset exit!')
        
    def get_data(self, corpus):
        source_data = []
        target_data = []
        for data in corpus:
            source_data.append(data['translation'][self.source_lang])
            target_data.append(data['translation'][self.target_lang])
        return source_data, target_data

    def save_data(self, data, save_path):
        print('=> Save data => Path: {}'.format(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data))