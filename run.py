import asyncio

import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import  CommandStart, StateFilter
from aiogram.fsm.state import default_state
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message,
    FSInputFile
)
from aiogram.utils.deep_linking import create_start_link

import config

from bot.states import UserStates
from bot.texts import *
from api import ParamsApi

TOKEN = config.bot_token

storage = MemoryStorage()
dp = Dispatcher(storage=storage)
bot = Bot(TOKEN)

api = ParamsApi()

bot_logger = logging.getLogger('bot_logger')

# Настраиваем формат и уровень логирования только для нашего логгера
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(name)s - %(message)s')
handler.setFormatter(formatter)
bot_logger.addHandler(handler)
bot_logger.setLevel(logging.INFO)

@dp.message(CommandStart())
async def command_start_handler(message: Message, state: FSMContext) -> None:
    bot_logger.info('Начало итерации')
    data = await state.get_data()
    try:
        message_id = data["delete_messege"]
        await bot.delete_messages(chat_id=message.chat.id, message_ids=message_id)
    except KeyError:
        await message.answer(hello_message_text)
    await message.answer(start_message_text)
    message_excel = await message.answer(pdf_message_text)
    user_id = message.from_user.id
    await state.update_data(delete_messege=[message_excel.message_id], user_id=user_id)
    await state.set_state(UserStates.get_pdf)


@dp.message(UserStates.get_pdf, F.content_type == "document")
async def get_pdf_handler(message: Message, state: FSMContext):
    bot_logger.info('Начало работы с документом')
    user_data = await state.get_data()
    message_id = user_data["delete_messege"]
    user_id = user_data["user_id"]
    await bot.delete_messages(chat_id=message.chat.id, message_ids=message_id)

    file_id = message.document.file_id
    file_name = f"{str(user_id)}_{message.document.file_name}"
    await state.update_data(file_name=file_name)

    file = await bot.get_file(file_id)
    file_path = file.file_path
    bot_logger.info(f'Документ "{file_name}" получен от пользователя {user_id}')
    await bot.download_file(file_path, f"files/{file_name}")
    messege_id = message.message_id
    await state.update_data(delete_messege=[messege_id + 1])

    waiting_message_id = await message.answer(waiting_message)
    try:
        answer = api.get_answer(f"files/{file_name}")

        await bot.delete_messages(chat_id=message.chat.id, message_ids=[waiting_message_id.message_id])
        
        await message.answer_document(FSInputFile(answer))
        message_after = await message.answer(
            "Что бы запусть бота заново напишите /start"
        )
        bot_logger.info('Конец итерации')
        await state.update_data(delete_messege=[message_after.message_id])
        await state.clear()
    except TypeError as e:
        await bot.delete_messages(chat_id=message.chat.id, message_ids=[waiting_message_id.message_id])
        await message.answer('Что-то пошло не по плану(')
        print(e)


@dp.message(StateFilter(UserStates.get_pdf), F.content_type != "document")
async def warning_not_pdf(message: Message, state: FSMContext):
    data = await state.get_data()
    message_id = data["delete_messege"]
    message_id.append(message.message_id - 1)

    await bot.delete_messages(chat_id=message.chat.id, message_ids=message_id)
    answer_text = f"{warning_pdf_message}\n\n{pdf_message_text}"
    await message.answer(text=answer_text)
    messege_id = message.message_id
    await state.update_data(delete_messege=[messege_id, messege_id + 1])
    data = await state.get_data()
    message_id = data["delete_messege"]


if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))