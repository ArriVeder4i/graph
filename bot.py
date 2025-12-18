import asyncio
import os
import io
import contextlib
import importlib
import MaximiN
import min2
import monoton
import planar

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
from prufer_draw import prufer_to_graph_file

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
GRAPH_FOLDER = os.getenv("GRAPH_FOLDER", "graphs")
os.makedirs(GRAPH_FOLDER, exist_ok=True)


class GraphStates(StatesGroup):
    choose_action = State()
    choose_mode = State()
    choose_file = State()
    enter_prufer = State()


bot = Bot(BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


@dp.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext):
    # Исправлено: убраны лишние кнопки и добавлена запятая
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="1️⃣ Посчитать готовую нумерацию", callback_data="action_count")],
            [InlineKeyboardButton(text="2️⃣ Найти оптимальную", callback_data="action_optimize")],
            [InlineKeyboardButton(text="3️⃣ Код Прюфера", callback_data="action_prufer")]
        ]
    )
    await message.answer("👋 Привет! Выбери действие:", reply_markup=keyboard)
    await state.set_state(GraphStates.choose_action)


@dp.callback_query(GraphStates.choose_action)
async def choose_action_callback(callback: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()

    if callback.data == "action_count_prufer":
        await state.update_data(action="count")
        await process_graph(callback.message, state, data.get("selected_file"))
        return

    if callback.data == "action_optimize_prufer":
        await state.update_data(action="optimize")
        # Кнопки после Прюфера
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="1️⃣ Maximin / Minimax", callback_data="mode_maximin_prufer")],
                [InlineKeyboardButton(text="2️⃣ Min / Max (Annealing)", callback_data="mode_min_prufer")],
                [InlineKeyboardButton(text="3️⃣ Монотонная (DAG)", callback_data="mode_monotone_prufer")],
                [InlineKeyboardButton(text="4️⃣ Плоская (Planar)", callback_data="mode_planar_prufer")]
            ]
        )
        await callback.message.answer("⚙️ Режим оптимизации для дерева Прюфера:", reply_markup=keyboard)
        await state.set_state(GraphStates.choose_mode)
        return

    if callback.data == "action_count":
        await state.update_data(action="count")
        await send_file_list(callback.message, state)
        await state.set_state(GraphStates.choose_file)
    elif callback.data == "action_optimize":
        await state.update_data(action="optimize")
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="1️⃣ Maximin / Minimax", callback_data="mode_maximin")],
                [InlineKeyboardButton(text="2️⃣ Min / Max (Annealing)", callback_data="mode_min")],
                [InlineKeyboardButton(text="3️⃣ Монотонная (DAG)", callback_data="mode_monotone")],
                [InlineKeyboardButton(text="4️⃣ Плоская (Planar)", callback_data="mode_planar")]
            ]
        )
        await callback.message.answer("⚙️ Режим оптимизации:", reply_markup=keyboard)
        await state.set_state(GraphStates.choose_mode)
    elif callback.data == "action_prufer":
        await callback.message.answer("Введите код Прюфера через пробел:")
        await state.set_state(GraphStates.enter_prufer)


@dp.callback_query(GraphStates.choose_mode)
async def choose_mode_callback(callback: types.CallbackQuery, state: FSMContext):
    mode_map = {
        "mode_maximin": "maximin", "mode_min": "min", "mode_monotone": "monotone", "mode_planar": "planar",
        "mode_maximin_prufer": "maximin", "mode_min_prufer": "min", "mode_monotone_prufer": "monotone",
        "mode_planar_prufer": "planar"
    }
    selected_mode = mode_map.get(callback.data)
    if not selected_mode: return
    await state.update_data(mode=selected_mode)
    if "_prufer" in callback.data:
        data = await state.get_data()
        await process_graph(callback.message, state, data.get("selected_file"))
    else:
        await send_file_list(callback.message, state)
        await state.set_state(GraphStates.choose_file)


async def send_file_list(message: types.Message, state: FSMContext):
    files = [f for f in os.listdir(GRAPH_FOLDER) if f.endswith(".graph")]
    buttons = [[InlineKeyboardButton(text=f, callback_data=f"file_{f}")] for f in files]
    buttons.append([InlineKeyboardButton(text="📎 Загрузить новый", callback_data="upload_new")])
    await message.answer("📂 Выбери файл:", reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons))


@dp.callback_query(GraphStates.choose_file)
async def choose_file_callback(callback: types.CallbackQuery, state: FSMContext):
    if callback.data == "upload_new":
        await callback.message.answer("Отправь .graph файл")
    else:
        path = os.path.join(GRAPH_FOLDER, callback.data.replace("file_", ""))
        await process_graph(callback.message, state, path)


@dp.message(GraphStates.enter_prufer)
async def handle_prufer(message: types.Message, state: FSMContext):
    try:
        prufer = list(map(int, message.text.split()))
        output_file = os.path.join(GRAPH_FOLDER, "prufer_generated.graph")
        prufer_to_graph_file(prufer, output_file)
        await state.update_data(selected_file=output_file)
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="1️⃣ Посчитать готовую", callback_data="action_count_prufer")],
                [InlineKeyboardButton(text="2️⃣ Найти оптимальную", callback_data="action_optimize_prufer")]
            ]
        )
        await message.answer("✅ Граф создан!", reply_markup=keyboard)
        await state.set_state(GraphStates.choose_action)
    except:
        await message.answer("❌ Ошибка в коде Прюфера.")


async def process_graph(message: types.Message, state: FSMContext, file_path: str):
    data = await state.get_data()
    action, mode = data.get("action"), data.get("mode")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            if action == "count":
                importlib.reload(min2);
                min2.main(file_path, choice="1")
            elif mode == "maximin":
                importlib.reload(MaximiN);
                MaximiN.main(file_path, choice="2")
            elif mode == "monotone":
                importlib.reload(monoton);
                monoton.main(file_path)
            elif mode == "planar":
                importlib.reload(planar);
                planar.main(file_path)
            else:
                importlib.reload(min2);
                min2.main(file_path, choice="2")
        except Exception as e:
            await message.answer(f"❌ Ошибка: {e}");
            return

    output = buffer.getvalue()
    if len(output) > 4000:
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(output)
        await message.answer_document(FSInputFile("output.txt"))
    else:
        await message.answer(f"📊 Результаты:\n\n{output[:4000]}")

    images = {"count": ["graph_from_file.png"], "maximin": ["graph_maximin.png", "graph_minimax.png"],
              "monotone": ["graph_monotone.png"], "planar": ["graph_planar.png"]}.get(
        mode if action != "count" else "count", ["graph_min.png", "graph_max.png"])
    for img in images:
        if os.path.exists(img): await message.answer_photo(FSInputFile(img))
    await state.clear();
    await cmd_start(message, state)


if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))