import logging
from src.data.loader import SLUDataLoader
from src.data.splitter import SLUDataSplitter
from src.data.feature_engineer import SLUFeatureEngineer
from src.data.builder import SLUDataBuilder
from src.data.data_utils import build_vocab, save_vocab

class DataPipeline:
    def __init__(self, config, data_dir, save_dir):
        self.config = config
        self.data_dir = data_dir
        self.save_dir = save_dir

    def prepare_data(self):
        logging.info("Loading and splitting data...")
        loader = SLUDataLoader(data_dir=self.data_dir)
        df = loader.load_all_data()
        
        splitter = SLUDataSplitter(df=df, test_size=self.config['test_size'], 
                                   random_state=self.config['random_state'])
        train_df, val_df = splitter.split_data()
        
        # Intent & Slot Feature Engineering
        feature_engineer = SLUFeatureEngineer(df, train_df, val_df)
        outputs = feature_engineer.engineer_features()
        
        # Build Vocabs
        word_to_id, id_to_word = build_vocab(train_df)
        intent_to_id = {label: idx for idx, label in enumerate(outputs[4].classes_)}
        
        # Mapping for slots with PAD
        slot_label_to_id = outputs[5]
        full_slot_mapping = {'<PAD>': 0, **{k: v+1 for k, v in slot_label_to_id.items()}}
        
        # Save assets
        save_vocab(word_to_id, f"{self.save_dir}/word_to_id.json")
        save_vocab(full_slot_mapping, f"{self.save_dir}/full_slot_mapping.json")
        save_vocab(intent_to_id, f"{self.save_dir}/intent_to_id.json")
        
        return train_df, val_df, word_to_id, full_slot_mapping, intent_to_id, outputs

    def build_slot_dataset(self, train_df, val_df):
        data_builder = SLUDataBuilder(train_df, val_df)
        X_train_slot, y_train_slot, _ = data_builder.build_crf_dataset(train_df)
        X_val_slot, y_val_slot, _ = data_builder.build_crf_dataset(val_df)
        return X_train_slot, y_train_slot, X_val_slot, y_val_slot