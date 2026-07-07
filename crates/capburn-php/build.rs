fn main() {
    // ext-php-rs 0.15+ сам находит конфигурацию PHP (include/link) в своём
    // build-скрипте. Здесь только пробрасываем версию сборки и настраиваем
    // линковку под macOS.
    println!("cargo:rerun-if-env-changed=PHP_CONFIG");
    println!("cargo:rerun-if-env-changed=PHP");

    let version = std::env::var("CAPBURN_PHP_VERSION")
        .ok()
        .map(|v| v.trim().trim_start_matches('v').to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "0.0.0-dev".to_string());
    println!("cargo:rustc-env=CAPBURN_PHP_BUILD_VERSION={version}");
    println!("cargo:rerun-if-env-changed=CAPBURN_PHP_VERSION");

    // На macOS PHP-расширение — это cdylib, ссылающийся на символы PHP,
    // которые доступны только когда его загрузит процесс PHP. Просим линковщик
    // не разрешать эти символы на этапе сборки, а связать их динамически при
    // загрузке.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
