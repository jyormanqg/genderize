from tensorflow.keras.models import load_model

import pandas as pd

import numpy as np

import unidecode

import os

import re

 

from pyspark.sql import functions as F

 

class LatamGenderize:

    def __init__(self, path=None):

        self.__pathDir = "." #os.path.join(os.path.dirname(os.path.abspath(__file__)))

        if path:

            self.__model   = self.__load_model(path)

        else:

            self.__model   = self.__load_model(os.path.join(self.__pathDir,'static/models/boyorgirl_CO_ES.h5'))

 

    def genderize(self, df, name_column = None):

        if not name_column:

            name_column = self.__identify_column_name(df.columns.tolist())

       

        df_trf = self.__preprocess(df, name_column)

        df_res = self.__predict(df_trf)

        df_res.drop(['clean_name_nlp','transform_name_nlp'], axis = 1, inplace = True)

        return df_res

 

    def __identify_column_name(self, src_cols):

        trf_cols = [col.lower() for col in src_cols]

        if 'name' in trf_cols:

            index = trf_cols.index('name')

        elif 'nombre' in trf_cols:

            index = trf_cols.index('nombre')

        return src_cols[index]

 

    def __load_model(self, modelPath):

        try:

            return load_model(modelPath)

        except Exception as e:

            raise e

 

    def __preprocess(self, df, name_column):

        serie_names = df[name_column].copy()

        clean_name  = serie_names.apply(lambda x: re.sub(r"[^a-z0-9 ]+", "", unidecode.unidecode(x.lower())))

        trf_names   = [list(name) for name in clean_name]

        name_length = 50

        trf_names   = [(name + [' '] * name_length)[:name_length] for name in trf_names]    # Pad names with spaces to make all names same length

        trf_names   = [[max(0.0, ord(char) - 96.0) for char in name] for name in trf_names] # Encode characters to number

 

        df['clean_name_nlp']     = clean_name

        df['transform_name_nlp'] = trf_names

        return df

 

    def __predict(self, df):

        result = self.__model.predict(np.asarray(df['transform_name_nlp'].values.tolist())).squeeze(axis=1)

        df['gender_predicted']   = ['M' if logit > 0.5 else 'F' for logit in result]

        df['gender_probability'] = [logit if logit > 0.5 else 1.0 - logit for logit in result]

        df['gender_probability'] = df['gender_probability'].round(2)

        return df