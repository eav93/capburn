<?php

// Стабы для расширения capburn_php (для автодополнения в IDE).
// Само распознавание выполняет нативное расширение.

namespace Capburn\Ext {
    /**
     * Распознаватель капчи: загружает обученную модель и предсказывает текст.
     */
    class Recognizer
    {
        /**
         * Загрузить модель из папки артефактов (`model.json` + `model.mpk`).
         *
         * @param string $artifactsDir путь к папке с обученной моделью
         */
        public function __construct(string $artifactsDir) {}

        /**
         * Распознать капчу из файла. Возвращает строку символов.
         *
         * @param string $imagePath путь к файлу изображения
         * @return string
         */
        public function recognize(string $imagePath): string {}

        /**
         * Распознать капчу из бинарной строки с содержимым изображения
         * (например результат `file_get_contents()` или тела HTTP-ответа).
         *
         * @param string $data сырые байты изображения (PNG/JPEG/…)
         * @return string
         */
        public function recognizeBytes(string $data): string {}

        /**
         * Распознать капчу из строки base64. Поддерживается «сырой» base64 и
         * data-URL вида `data:image/png;base64,iVBORw0...`.
         *
         * @param string $data base64-строка или data-URL
         * @return string
         */
        public function recognizeBase64(string $data): string {}

        /**
         * Длина капчи (число символов), которое возвращает модель.
         *
         * @return int
         */
        public function numChars(): int {}

        /**
         * Версия сборки расширения (тег релиза либо `0.0.0-dev`).
         *
         * @return string
         */
        public static function extensionVersion(): string {}
    }
}
