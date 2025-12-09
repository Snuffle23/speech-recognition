import whisper
import torch
import os
import time
from pathlib import Path
from datetime import datetime
import re
import sys
import warnings
import logging
warnings.filterwarnings("ignore")

# Настройка логирования
logging.basicConfig(
    filename='whisper_processor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

print("="*70)
print("WHISPER BATCH AUDIO PROCESSOR")
print("="*70)

# Очистка памяти GPU в начале
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Проверка и установка недостающих зависимостей
def check_dependencies():
    """Проверка и установка необходимых зависимостей"""
    missing_deps = []
    
    # Проверка основных библиотек
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import whisper
    except ImportError:
        missing_deps.append("openai-whisper")
    
    try:
        import mutagen
    except ImportError:
        missing_deps.append("mutagen")
    
    if missing_deps:
        print(f"Обнаружены отсутствующие зависимости: {', '.join(missing_deps)}")
        print("Установите их командой: pip install " + " ".join(missing_deps))
        response = input("Установить автоматически? (y/n): ").lower()
        if response == 'y':
            import subprocess
            for dep in missing_deps:
                print(f"Установка {dep}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print("Зависимости установлены. Перезапустите скрипт.")
            sys.exit(0)
        else:
            print("Продолжение невозможно без установленных зависимостей.")
            sys.exit(1)
    
    print("Все зависимости установлены.")

# Проверяем зависимости
check_dependencies()

# Импортируем дополнительные библиотеки
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Примечание: Установите tqdm для прогресс-бара: pip install tqdm")

try:
    import humanize
    HAS_HUMANIZE = True
except ImportError:
    HAS_HUMANIZE = False

# Определение устройства
if torch.cuda.is_available():
    device = "cuda"
    print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
    print(f"GPU память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} ГБ")
    # Очистка памяти перед началом работы
    torch.cuda.empty_cache()
else:
    device = "cpu"
    print("Используется CPU (GPU не обнаружен)")

print("="*70)

class AdvancedWhisperProcessor:
    def __init__(self, model_name="small", language=None):
        """
        Инициализация улучшенного процессора
        
        Параметры:
        ----------
        model_name : str
            Модель Whisper (tiny, base, small, medium, large, large-v3)
            По умолчанию 'small' - лучший баланс для русского языка
        language : str или None
            Язык аудио (например, 'ru', 'en'), если None - автоопределение
        """
        self.model_name = model_name
        self.language = language
        self.device = device
        
        print(f"Загрузка модели '{model_name}' на устройстве {device}...")
        logging.info(f"Загрузка модели '{model_name}' на устройстве {device}")
        start_time = time.time()
        
        try:
            self.model = whisper.load_model(model_name, device=device)
            load_time = time.time() - start_time
            print(f"Модель загружена за {load_time:.2f} секунд")
            logging.info(f"Модель '{model_name}' загружена за {load_time:.2f} секунд")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            logging.error(f"Ошибка загрузки модели: {e}")
            print("Рекомендации:")
            print("1. Проверьте доступную память")
            print("2. Используйте меньшую модель (tiny или base)")
            print("3. Перезапустите скрипт")
            sys.exit(1)
            
        """
        # Настройки транскрибации (оптимизированы для стабильности)
        self.transcribe_options = {
            "task": "transcribe",
            "language": language,
            "verbose": False,
            "fp16": True if device == "cuda" else False,
            "temperature": 0.0,  # Детерминированность
            "best_of": 3,  # Уменьшено для стабильности
            "beam_size": 3,  # Уменьшено для стабильности
            "patience": None,  # Отключено для предотвращения зацикливания
            "compression_ratio_threshold": 2.2,  # Более строгий порог
            "no_speech_threshold": 0.7,  # Более строгий порог
            "condition_on_previous_text": False,  # Отключено для стабильности
            "initial_prompt": None,
            "word_timestamps": True,  # Включено для лучшей сегментации
            "suppress_tokens": "-1",  # Подавление пустых токенов
        }
        """

        # Настройки транскрибации (оптимизированы для эха)
        self.transcribe_options = {
            "task": "transcribe",
            "language": language,
            "verbose": False,
            "fp16": True if device == "cuda" else False,
            "temperature": 0.0,  # Фиксированная температура
            "best_of": 5,       # Увеличить для шумного аудио
            "beam_size": 5,     # Увеличить для шумного аудио
            "patience": None,   # Отключить для предотвращения зацикливания
            "compression_ratio_threshold": 2.6,  # Увеличить порог
            "no_speech_threshold": 0.5,  # Уменьшить порог (0.5 вместо 0.6)
            "condition_on_previous_text": False,  # Отключить для шумного аудио
            "initial_prompt": "Привет, это аудио с эхом и шумами",  # Контекстная подсказка
            "word_timestamps": False,  # Отключить для шумного аудио
            "suppress_tokens": "-1",
            "logprob_threshold": -0.8,  # Порог логической вероятности
            "no_repeat_ngram_size": 3,  # Предотвращает повторения
}
        
        # Упрощенные параметры для повторной попытки
        self.simple_transcribe_options = {
            "task": "transcribe",
            "language": language,
            "verbose": False,
            "fp16": True if device == "cuda" else False,
            "temperature": 0.0,
            "best_of": 1,
            "beam_size": 1,
            "word_timestamps": False,
        }
    
    def sanitize_filename(self, filename):
        """Очистка имени файла от недопустимых символов"""
        # Заменяем недопустимые символы в Windows
        invalid_chars = r'[<>:"/\\|?*]'
        filename = re.sub(invalid_chars, '_', filename)
        
        # Убираем точки в конце имени
        filename = filename.rstrip('.')
        
        # Ограничиваем длину имени
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200] + ext
        
        return filename
    
    def ensure_directory(self, path):
        """Создание директории если её нет"""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                return True
            except Exception as e:
                print(f"Не удалось создать папку {directory}: {e}")
                logging.error(f"Не удалось создать папку {directory}: {e}")
                return False
        return True
    
    def get_audio_info(self, audio_path):
        """Получение информации об аудиофайле"""
        try:
            from mutagen import File
            
            file_size = os.path.getsize(audio_path) / (1024 * 1024)  # МБ
            
            # Получаем длительность
            try:
                audio = File(audio_path)
                if audio and audio.info:
                    duration = audio.info.length
                else:
                    audio_data = whisper.load_audio(audio_path)
                    duration = len(audio_data) / 16000
            except:
                audio_data = whisper.load_audio(audio_path)
                duration = len(audio_data) / 16000
            
            # Определение формата
            format_name = Path(audio_path).suffix[1:].upper()
            
            return {
                'duration': duration,
                'size_mb': file_size,
                'format': format_name
            }
            
        except Exception as e:
            print(f"Не удалось получить информацию об аудио: {e}")
            logging.warning(f"Не удалось получить информацию об аудио {audio_path}: {e}")
            return None
    
    def format_time(self, seconds):
        """Форматирование времени в ЧЧ:ММ:СС.ммм"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}"
    
    def save_results(self, result, input_path, output_path, audio_duration, processing_time):
        """Сохранение результатов в файл"""
        try:
            # Создаем папку если её нет
            if not self.ensure_directory(output_path):
                # Если не удалось создать папку, сохраняем в текущую директорию
                output_path = os.path.basename(output_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Заголовок
                f.write(f"Транскрибация аудио: {Path(input_path).name}\n")
                f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Модель: {self.model_name}\n")
                f.write(f"Устройство: {self.device}\n")
                f.write(f"Язык: {result.get('language', 'auto')}\n")
                f.write(f"Длительность: {audio_duration:.2f} сек\n")
                f.write(f"Время обработки: {processing_time:.2f} сек\n")
                f.write("=" * 80 + "\n\n")
                
                # Полный текст
                f.write("ПОЛНЫЙ ТЕКСТ:\n")
                f.write("=" * 80 + "\n")
                f.write(result["text"])
                f.write("\n\n" + "=" * 80 + "\n\n")
                
                # Текст с временными метками
                f.write("ТЕКСТ С ВРЕМЕННЫМИ МЕТКАМИ:\n")
                f.write("=" * 80 + "\n")
                
                for i, segment in enumerate(result["segments"], 1):
                    start_time_str = self.format_time(segment["start"])
                    end_time_str = self.format_time(segment["end"])
                    
                    f.write(f"[{start_time_str} --> {end_time_str}] ")
                    f.write(segment["text"].strip() + "\n")
                
                # Статистика
                f.write("\n" + "=" * 80 + "\n")
                f.write("СТАТИСТИКА:\n")
                f.write(f"Всего сегментов: {len(result['segments'])}\n")
                
                words_count = len(result["text"].split())
                f.write(f"Примерное количество слов: {words_count}\n")
                
                if audio_duration > 0:
                    words_per_minute = (words_count / audio_duration) * 60
                    f.write(f"Скорость речи: {words_per_minute:.1f} слов/минуту\n")
            
            print(f"Результаты сохранены в: {output_path}")
            logging.info(f"Результаты сохранены в: {output_path}")
            return True
            
        except Exception as e:
            print(f"Ошибка при сохранении файла {output_path}: {e}")
            logging.error(f"Ошибка при сохранении файла {output_path}: {e}")
            
            # Попытка сохранить с альтернативным именем
            try:
                alt_path = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(alt_path, 'w', encoding='utf-8') as f:
                    f.write(result["text"])
                print(f"Сохранено с альтернативным именем: {alt_path}")
                logging.info(f"Сохранено с альтернативным именем: {alt_path}")
                return True
            except:
                print("Не удалось сохранить файл. Текст транскрибации:")
                print("-" * 50)
                print(result["text"][:1000])
                print("..." if len(result["text"]) > 1000 else "")
                return False
    
    def transcribe_audio(self, audio_path, use_simple_params=False):
        """Транскрибация аудио с обработкой ошибок"""
        try:
            if use_simple_params:
                print("   Используются упрощенные параметры для стабильности")
                result = self.model.transcribe(
                    audio=audio_path,
                    **self.simple_transcribe_options
                )
            else:
                result = self.model.transcribe(
                    audio=audio_path,
                    **self.transcribe_options
                )
            return result, None
        except Exception as e:
            error_msg = f"Ошибка транскрибации: {str(e)}"
            print(f"   {error_msg}")
            logging.error(f"Ошибка транскрибации {audio_path}: {error_msg}")
            return None, str(e)
    
    def process_audio_file(self, input_path, output_path, use_chunking=False, chunk_minutes=1):
        """
        Обработка одного аудиофайла
        
        Параметры:
        ----------
        input_path : str
            Путь к входному аудиофайлу
        output_path : str
            Путь для сохранения текстового файла
        use_chunking : bool
            Обрабатывать ли файл по частям
        chunk_minutes : int
            Длительность одной части в минутах
        """
        try:
            print(f"\nОбработка файла: {Path(input_path).name}")
            logging.info(f"Начало обработки файла: {input_path}")
            start_time = time.time()
            
            # Проверка существования файла
            if not os.path.exists(input_path):
                error_msg = f"Файл не найден: {input_path}"
                print(f"{error_msg}")
                logging.error(error_msg)
                return None
            
            # Получение информации об аудио
            audio_info = self.get_audio_info(input_path)
            if audio_info:
                print(f"   Длительность: {audio_info['duration']:.2f} сек ({audio_info['duration']/60:.2f} мин)")
                print(f"   Размер: {audio_info['size_mb']:.2f} МБ")
                print(f"   Формат: {audio_info['format']}")
            
            # Проверка необходимости обработки по частям
            audio_duration = audio_info['duration'] if audio_info else 0
            use_chunking = use_chunking or (audio_duration > 300)  # Более 5 минут
            
            if use_chunking and audio_duration > 60:  # Более 1 минут
                print(f"   Файл длинный ({audio_duration/60:.1f} мин), используется обработка по частям")
                return self.process_long_audio(input_path, output_path, chunk_minutes)
            
            # Транскрибация
            print(f"   Начало транскрибации...")
            result, error = self.transcribe_audio(input_path, use_simple_params=False)
            
            # Если ошибка, пробуем упрощенные параметры
            if error:
                print(f"   Повторная попытка с упрощенными параметрами...")
                result, error = self.transcribe_audio(input_path, use_simple_params=True)
            
            if error or not result:
                error_msg = f"Не удалось транскрибировать файл: {error}"
                print(f"   {error_msg}")
                logging.error(f"Не удалось транскрибировать файл {input_path}: {error_msg}")
                return {
                    'success': False,
                    'input_file': input_path,
                    'error': error_msg
                }
            
            # Проверка на зацикливание (повторяющийся текст)
            text = result["text"].strip()
            if len(text) > 100:
                # Проверяем первые 100 символов на повторение
                sample = text[:100]
                repetitions = text.count(sample[:20])  # Ищем повторение первых 20 символов
                if repetitions > 3:
                    print(f"   Обнаружено зацикливание текста. Повторная попытка с упрощенными параметрами...")
                    result, error = self.transcribe_audio(input_path, use_simple_params=True)
                    if error or not result:
                        return {
                            'success': False,
                            'input_file': input_path,
                            'error': "Зацикливание текста"
                        }
            
            # Расчет времени
            processing_time = time.time() - start_time
            
            # Сохранение результатов
            audio_duration = audio_info['duration'] if audio_info else 0
            save_success = self.save_results(result, input_path, output_path, audio_duration, processing_time)
            
            if save_success:
                speed_ratio = processing_time / audio_duration if audio_duration > 0 else 0
                
                print(f"   Успешно обработан за {processing_time:.2f} сек")
                print(f"   Скорость обработки: {speed_ratio:.3f}x реального времени")
                logging.info(f"Файл {input_path} успешно обработан за {processing_time:.2f} сек")
                
                return {
                    'success': True,
                    'input_file': input_path,
                    'output_file': output_path,
                    'processing_time': processing_time,
                    'text_length': len(result['text']),
                    'segments': len(result['segments']),
                    'language': result.get('language', 'unknown'),
                    'audio_duration': audio_duration
                }
            else:
                return {
                    'success': False,
                    'input_file': input_path,
                    'error': 'Ошибка сохранения файла'
                }
            
        except Exception as e:
            error_msg = f"Ошибка при обработке {input_path}: {str(e)}"
            print(f"{error_msg}")
            logging.error(error_msg)
            return {
                'success': False,
                'input_file': input_path,
                'error': str(e)
            }
    
    def find_audio_files(self, input_folder, extensions=None):
        """Находит аудиофайлы в папке без дубликатов"""
        if extensions is None:
            extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.m4b']
        
        # Находим все файлы с заданными расширениями
        all_files = []
        for ext in extensions:
            # Ищем в нижнем регистре
            all_files.extend(Path(input_folder).glob(f"*{ext.lower()}"))
            # Ищем в верхнем регистре
            all_files.extend(Path(input_folder).glob(f"*{ext.upper()}"))
        
        if not all_files:
            return []
        
        # Удаляем дубликаты (одинаковые имена файлов в разном регистре)
        seen = set()
        unique_files = []
        for file_path in all_files:
            # Используем нижний регистр для сравнения
            file_key = str(file_path).lower()
            if file_key not in seen:
                seen.add(file_key)
                unique_files.append(file_path)
        
        # Сортируем файлы по размеру для более предсказуемого времени обработки
        unique_files.sort(key=lambda x: x.stat().st_size if x.exists() else 0)
        
        return unique_files
    
    def process_folder(self, input_folder, output_folder, extensions=None, chunk_minutes=1):
        """
        Обработка всех аудиофайлов в папке
        
        Параметры:
        ----------
        input_folder : str
            Папка с аудиофайлами
        output_folder : str
            Папка для сохранения результатов
        extensions : list
            Список расширений файлов для обработки
        chunk_minutes : int
            Длительность одной части для длинных файлов
        """
        # Создание выходной папки
        os.makedirs(output_folder, exist_ok=True)
        
        # Поиск аудиофайлов без дубликатов
        audio_files = self.find_audio_files(input_folder, extensions)
        
        if not audio_files:
            print(f"Не найдено аудиофайлов в папке {input_folder}")
            logging.warning(f"Не найдено аудиофайлов в папке {input_folder}")
            return []
        
        print(f"Найдено {len(audio_files)} аудиофайлов")
        print(f"Уникальные файлы:")
        for i, file_path in enumerate(audio_files, 1):
            print(f"  {i}. {file_path.name}")
        
        # Обработка файлов
        results = []
        
        # Используем tqdm если установлен, иначе обычный цикл
        if HAS_TQDM:
            if HAS_TQDM:
                iterator = tqdm(
                    audio_files, 
                    desc="Обработка файлов",
                    ncols=20,  # Ширина прогресс-бара
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
        else:
            iterator = audio_files
        
        for i, audio_file in enumerate(iterator, 1):
            if not HAS_TQDM:
                print(f"\n" + "="*60)
                print(f"Файл {i}/{len(audio_files)}: {audio_file.name}")
            
            # Проверяем длительность для решения об обработке по частям
            audio_info = self.get_audio_info(str(audio_file))
            use_chunking = audio_info and audio_info['duration'] > 300  # Более 5 минут
            
            # Создание безопасного имени выходного файла
            safe_name = self.sanitize_filename(audio_file.stem)
            output_filename = f"{safe_name}_transcript.txt"
            output_path = os.path.join(output_folder, output_filename)
            
            # Обработка файла
            result = self.process_audio_file(
                str(audio_file), 
                output_path, 
                use_chunking=use_chunking,
                chunk_minutes=chunk_minutes
            )
            results.append(result)
            
            # Очистка памяти GPU после каждого файла
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # Вывод статистики
        self.print_statistics(results)
        
        return results
    
    def print_statistics(self, results):
        """Вывод статистики по обработке"""
        successful = [r for r in results if r and r['success']]
        failed = [r for r in results if r and not r['success']]
        
        print(f"\n" + "="*70)
        print("СТАТИСТИКА ОБРАБОТКИ")
        print("="*70)
        
        print(f"Успешно обработано: {len(successful)} файлов")
        print(f"Не удалось обработать: {len(failed)} файлов")
        logging.info(f"Обработка завершена: успешно {len(successful)} файлов, ошибок {len(failed)}")
        
        if successful:
            total_time = sum(r['processing_time'] for r in successful)
            avg_time = total_time / len(successful)
            total_text = sum(r['text_length'] for r in successful)
            total_audio = sum(r.get('audio_duration', 0) for r in successful)
            
            print(f"\nОбщее время обработки: {total_time:.2f} сек ({total_time/60:.1f} мин)")
            print(f"Среднее время на файл: {avg_time:.2f} сек")
            print(f"Общий объем текста: {total_text:,} символов")
            
            if total_audio > 0:
                speed_ratio = total_time / total_audio
                print(f"Общая длительность аудио: {total_audio/60:.1f} мин")
                print(f"Средняя скорость: {speed_ratio:.3f}x (в {1/speed_ratio:.1f} раз быстрее реального времени)")
            
            if HAS_HUMANIZE:
                import humanize
                total_size_mb = sum(r.get('size_mb', 0) for r in successful)
                print(f"Общий размер обработанных файлов: {humanize.naturalsize(total_size_mb*1024*1024)}")
        
        if failed:
            print(f"\nОшибки обработки:")
            logging.warning(f"Ошибки обработки ({len(failed)} файлов):")
            for fail in failed[:5]:  # Показываем только первые 5 ошибок
                error_msg = f"  {Path(fail['input_file']).name}: {fail.get('error', 'Unknown error')}"
                print(error_msg)
                logging.warning(error_msg)
            if len(failed) > 5:
                print(f"  ... и еще {len(failed)-5} ошибок")
                logging.warning(f"  ... и еще {len(failed)-5} ошибок")
        
        print("="*70)
    
    def process_long_audio(self, input_path, output_path, chunk_minutes=1):
        """
        Обработка очень длинных аудиофайлов (более 5 минут) по частям
        
        Параметры:
        ----------
        input_path : str
            Путь к длинному аудиофайлу
        output_path : str
            Путь для сохранения результата
        chunk_minutes : int
            Длительность одной части в минутах
        """
        try:
            # Проверяем наличие pydub
            try:
                from pydub import AudioSegment
            except ImportError:
                print("Библиотека pydub не установлена. Установите: pip install pydub")
                print("Используется обычная обработка (может быть нестабильной)...")
                return self.process_audio_file(input_path, output_path, use_chunking=False)
            
            import tempfile
            
            print(f"Разбиваю длинное аудио на части по {chunk_minutes} минут...")
            logging.info(f"Начало обработки длинного аудио {input_path} по частям")
            
            audio = AudioSegment.from_file(input_path)
            chunk_ms = chunk_minutes * 60 * 1000
            
            transcripts = []
            total_chunks = len(audio) // chunk_ms + (1 if len(audio) % chunk_ms > 0 else 0)
            
            print(f"Всего частей: {total_chunks}")
            
            for i in range(0, len(audio), chunk_ms):
                chunk_num = i // chunk_ms + 1
                chunk = audio[i:i + chunk_ms]
                
                print(f"  Обработка части {chunk_num}/{total_chunks}...")
                
                # Сохраняем временный файл
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    chunk.export(tmp.name, format="wav")
                    tmp_path = tmp.name
                
                try:
                    # Транскрибация части
                    result, error = self.transcribe_audio(tmp_path, use_simple_params=True)
                    
                    if error or not result:
                        print(f"  Ошибка в части {chunk_num}: {error}")
                        logging.warning(f"Ошибка в части {chunk_num} файла {input_path}: {error}")
                        transcripts.append(f"[Ошибка в части {chunk_num}]")
                    else:
                        transcripts.append(result["text"])
                finally:
                    # Удаляем временный файл
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                
                # Очистка памяти
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # Объединяем результаты
            full_text = ""
            for i, text in enumerate(transcripts, 1):
                start_min = (i-1) * chunk_minutes
                end_min = i * chunk_minutes
                full_text += f"\n[Часть {i}: {start_min:02d}:00 - {end_min:02d}:00]\n"
                full_text += text + "\n"
            
            # Сохраняем
            self.ensure_directory(output_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Длинное аудио (обработано по частям): {Path(input_path).name}\n")
                f.write(f"Всего частей: {total_chunks} по {chunk_minutes} минут\n")
                f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Модель: {self.model_name}\n")
                f.write(f"Устройство: {self.device}\n")
                f.write("="*80 + "\n")
                f.write(full_text)
            
            print(f"Длинное аудио успешно обработано и сохранено в: {output_path}")
            logging.info(f"Длинное аудио {input_path} успешно обработано, {total_chunks} частей")
            
            return {
                'success': True,
                'output_file': output_path,
                'chunks': total_chunks,
                'total_text': len(full_text)
            }
            
        except Exception as e:
            error_msg = f"Ошибка при обработке длинного аудио: {e}"
            print(error_msg)
            logging.error(f"Ошибка при обработке длинного аудио {input_path}: {e}")
            return {'success': False, 'error': str(e)}


def main():
    """Основная функция программы"""
    print("\nНастройка параметров обработки")
    print("-" * 40)
    
    # Определение папок
    input_folder = "audio_input"
    output_folder = "transcripts"
    
    # Создаем папки если их нет
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Папка с аудио: {os.path.abspath(input_folder)}")
    print(f"Папка для результатов: {os.path.abspath(output_folder)}")
    
    # Выбор модели
    print("\nДоступные модели:")
    models = [
        ("tiny", "Очень быстро, низкая точность"),
        ("base", "Баланс скорости и точности"),
        ("small", "Хорошая точность (рекомендуется для русского)"),
        ("medium", "Высокая точность, медленно"),
        ("large", "Наилучшая точность, очень медленно"),
    ]
    
    for i, (name, desc) in enumerate(models, 1):
        print(f"{i}. {name:6} - {desc}")
    
    model_choice = input(f"\nВыберите модель (1-5) [3 - рекомендуется]: ").strip()
    model_map = {1: "tiny", 2: "base", 3: "small", 4: "medium", 5: "large"}
    model_name = model_map.get(int(model_choice) if model_choice.isdigit() else 3, "small")
    
    # Выбор языка
    language = input("Введите код языка (ru, en, es, fr и т.д.) или оставьте пустым для автоопределения: ").strip()
    language = language if language else None
    
    # Параметры для длинных файлов
    print("\nНастройки для длинных аудиофайлов (>5 минут):")
    print("1. Автоматически разбивать на части (рекомендуется)")
    print("2. Обрабатывать целиком (может быть нестабильно)")
    chunk_choice = input("Выберите опцию (1-2) [1]: ").strip()
    use_chunking = chunk_choice != "2"
    
    chunk_minutes = 1
    if use_chunking:
        try:
            chunk_minutes = int(input(f"Длительность одной части в минутах [1]: ").strip() or "1")
        except ValueError:
            chunk_minutes = 1
    
    # Создание процессора
    processor = AdvancedWhisperProcessor(
        model_name=model_name,
        language=language
    )
    
    # Проверка наличия файлов
    audio_files = processor.find_audio_files(input_folder)
    
    if not audio_files:
        print(f"\nВ папке '{input_folder}' не найдено аудиофайлов!")
        print(f"Поддерживаемые форматы: .mp3, .wav, .m4a, .flac, .ogg, .aac, .wma, .m4b")
        print(f"Поместите аудиофайлы в папку и запустите программу снова.")
        input("\nНажмите Enter для выхода...")
        return
    
    print(f"\nНайдено {len(audio_files)} аудиофайлов для обработки")
    print("Файлы для обработки:")
    for i, file_path in enumerate(audio_files, 1):
        file_info = processor.get_audio_info(str(file_path))
        if file_info:
            duration_str = f"{file_info['duration']/60:.1f} мин"
        else:
            duration_str = "неизвестно"
        print(f"  {i}. {file_path.name} ({duration_str})")
    
    # Подтверждение
    print(f"\nПараметры обработки:")
    print(f"  Модель: {model_name}")
    print(f"  Устройство: {device}")
    print(f"  Язык: {language or 'автоопределение'}")
    print(f"  Обработка длинных файлов: {'по частям' if use_chunking else 'целиком'}")
    if use_chunking:
        print(f"  Длительность части: {chunk_minutes} минут")
    print(f"  Количество файлов: {len(audio_files)}")
    
    confirm = input("\nНачать обработку? (y/n): ").lower()
    if confirm != 'y':
        print("Обработка отменена.")
        return
    
    # Запуск обработки
    print("\n" + "="*70)
    print("НАЧАЛО ОБРАБОТКИ")
    print("="*70)
    
    start_time = time.time()
    
    results = processor.process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        chunk_minutes=chunk_minutes
    )
    
    total_time = time.time() - start_time
    
    print(f"\nОбработка завершена за {total_time:.2f} секунд ({total_time/60:.1f} минут)")
    print(f"Результаты сохранены в папке: {os.path.abspath(output_folder)}")
    print(f"Лог ошибок сохранен в файле: whisper_processor.log")
    
    # Очистка памяти GPU
    if device == "cuda":
        torch.cuda.empty_cache()
    
    input("\nНажмите Enter для выхода...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nОбработка прервана пользователем.")
        logging.info("Обработка прервана пользователем")
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        logging.critical(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        logging.critical(f"Traceback: {traceback.format_exc()}")
    
    input("\nНажмите Enter для завершения...")