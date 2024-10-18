#!/usr/bin/python
# -*- coding: utf8 -*-

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import re
import pandas as pd
import time
import logging
import xlsxwriter

from bot.texts import *

import config

api_logger = logging.getLogger("api_logger")

# Настраиваем формат и уровень логирования только для нашего логгера
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s:%(name)s - %(message)s")
handler.setFormatter(formatter)
api_logger.addHandler(handler)
api_logger.setLevel(logging.INFO)


class ParamsApi:
    def __init__(self, api_key=config.api_key, model="gpt-4o"):
        self.api_key = api_key
        self.model = model

        self.client = OpenAI(api_key=self.api_key)

    def split_doc(self, file_path) -> list:
        document = fitz.open(file_path)
        text = chr(12).join([page.get_text() for page in document])
        pattern = r"((?:\n{1}\d{1,2}(?:\.\d+)*\.? *[А-Я]?|[0-9]?)(?:[^\n]+)(?:\n(?!\d{1,2}(?:\.\d+)*\.? )[^\n]+)*)"
        matches = re.findall(pattern, text)
        api_logger.info("Документ разделен на чанки")
        return matches

    def candidate_selection(self, anchor_list, chanks):
        api_logger.info("Отбор кандидатов")
        first_timestemp = time.time()
        lists_of_candidats = []
        treshold_list = [
            0.34,
            0.44,
            0.48,
            0.48,
            0.49,
            0.41,
            0.51,
            0.45,
            0.35,
            0.49,
            0.43,
            0.46,
        ]
        chanks_embeddings = [
            self.client.embeddings.create(input=chank, model="text-embedding-3-large")
            .data[0]
            .embedding
            for chank in chanks
        ]
        for row in range(len(anchor_list)):
            anchor_embedding = (self.client.embeddings.create(input=anchor_list[row], model="text-embedding-3-large").data[0].embedding)
            scores = cosine_similarity([anchor_embedding], chanks_embeddings)[0]
            candits = [
                re.sub(r'\s+', ' ', chanks[score_id]).strip()
                for score_id, score in enumerate(scores)
                if score >= treshold_list[row]
            ]
            lists_of_candidats.append(candits)
        api_logger.info(
            f"Отбор кандидатов завершен.\n  На это ушло: {time.time() - first_timestemp} секунд"
        )
        return lists_of_candidats

    def category_finding(self, lists_of_candidats):
        answer_dict = {}
        prompt_token_sum = 0
        completion_token_sum = 0
        total_token_sum = 0
        first_timestemp = time.time()
        for candidat_id, candidats in enumerate(lists_of_candidats):
            candidat_timestemp = time.time()
            api_logger.info(f"Category_name {column_name_list[candidat_id]}")
            for candidat in candidats:
                messeges = [
                    {
                        "role": "system",
                        "content": system_prompt + discription_prompt_list[candidat_id],
                    }
                ]
                messeges.append({"role": "user", "content": candidat})
                response = self.client.chat.completions.create(
                    model=self.model, messages=messeges
                )
                answer = response.choices[0].message.content
                prompt_token_sum += response.usage.prompt_tokens
                completion_token_sum += response.usage.completion_tokens
                total_token_sum += response.usage.total_tokens
                if "True" in answer:
                    if column_name_list[candidat_id] not in answer_dict:
                        answer_dict[column_name_list[candidat_id]] = [candidat]
                        continue
                    elif column_name_list[candidat_id] in answer_dict:
                        answer_dict[column_name_list[candidat_id]].append(candidat)
                        continue
            api_logger.info(
            f"На обработку {column_name_list[candidat_id]} ушло: {time.time() - candidat_timestemp} секунд"
        )
        api_logger.info(
            f"На обработку всех категорий ушло: {time.time() - first_timestemp} секунд"
        )
        api_logger.info(f"Всего токенов потрачено {total_token_sum}")
        api_logger.info("Этап заполнения колонок завершен")
        return answer_dict

    def dict_to_excel(self, doc, path):
        new_doc = {}
        for column_name in column_name_list:
            if column_name in doc:
                column_text = " ".join(doc[column_name])
            elif column_name not in doc:
                column_text = "--------"
            new_doc[column_name] = [column_text]
        df = pd.DataFrame.from_dict(new_doc)
        df['file_name'] = path[6:]
        path = f"{path[:-4]}_catigoriens.xlsx"
        df.to_excel(path, index=False, engine='xlsxwriter')
        api_logger.info("Ответ запакован в xlsx")
        return path

    def get_answer(self, file_path):
        api_logger.info("Начало работы api")
        chanks = self.split_doc(file_path)
        list_of_candidats = self.candidate_selection(anchor_list, chanks)
        answer = self.category_finding(list_of_candidats)
        answer_df = self.dict_to_excel(answer, file_path)
        api_logger.info("Файл готов к отправке")
        return answer_df
