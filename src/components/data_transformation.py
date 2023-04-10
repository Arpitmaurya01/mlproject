import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exeption import CustomException
from src.logger import logging
from src.utills import save_object
import os

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_features =['reading_score', 'writing_score']

            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course',]

            num_pipeline=Pipeline(
                steps=[
                ("Imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler(with_mean=False))
                ]
                )

            cat_pipeline=Pipeline(
                steps=[
                ("Impueter",SimpleImputer(strategy="most_frequent")),
                ("Encoading",OneHotEncoder()),
                ("Scaler",StandardScaler(with_mean=False))
                ]


            )
            logging.info(f"categorical columns:{categorical_features}")
            logging.info(f"numerical columns:{numerical_features}")

            preprocessor=ColumnTransformer(
                [

                ("num_pipeline",num_pipeline,numerical_features),
                ("cat_pipeline",cat_pipeline,categorical_features)
                ]

            )
            return preprocessor
    
            
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transfer(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("test and train data imported")
            logging.info("obtaining preprocessing object ")

            preprocessing_obj=self.get_data_transformer_obj()
            

            target_column_name="math_score"
            numerical_columns =['reading_score', 'writing_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)

