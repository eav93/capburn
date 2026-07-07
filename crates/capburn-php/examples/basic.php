<?php
// Пример использования расширения capburn_php.
//
// Запуск (macOS/Linux), указав путь к собранному расширению и папке с моделью:
//   php -d extension=/path/to/libcapburn_php.dylib examples/basic.php ./artifacts ./captcha.png
//
// Обычно расширение прописывают в php.ini (extension=capburn_php.so),
// тогда флаг -d не нужен.

[$script, $artifactsDir, $imagePath] = $argv + [null, './artifacts', './captcha.png'];

if (!class_exists('Capburn\\Ext\\Recognizer')) {
    fwrite(STDERR, "Расширение capburn_php не загружено.\n");
    exit(1);
}

use Capburn\Ext\Recognizer;

echo 'Версия расширения: ' . Recognizer::extensionVersion() . PHP_EOL;

$recognizer = new Recognizer($artifactsDir);
echo 'Длина капчи: ' . $recognizer->numChars() . PHP_EOL;

// 1) Из файла
echo 'Из файла:   ' . $recognizer->recognize($imagePath) . PHP_EOL;

// 2) Из сырых байтов (например тело HTTP-ответа)
$bytes = file_get_contents($imagePath);
echo 'Из байтов:  ' . $recognizer->recognizeBytes($bytes) . PHP_EOL;

// 3) Из base64 (в том числе data-URL)
$b64 = base64_encode($bytes);
echo 'Из base64:  ' . $recognizer->recognizeBase64($b64) . PHP_EOL;
echo 'Из dataURL: ' . $recognizer->recognizeBase64('data:image/png;base64,' . $b64) . PHP_EOL;
