import sys
import traceback

def main() -> int:
    try:
        import torch
    except Exception as e:
        print("❌ Не удалось импортировать torch:", repr(e))
        return 1

    print("=== PyTorch / CUDA environment check ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA compiled (torch.version.cuda): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

    # Базовая доступность CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available (torch.cuda.is_available): {cuda_available}")

    # Если CUDA недоступна — завершаем с подсказками
    if not cuda_available:
        print("\n⚠️ CUDA не доступна.")
        print("Возможные причины:")
        print("- У тебя нет NVIDIA GPU или ты не в среде с проброшенной GPU (WSL без GPU, контейнер без --gpus, etc.)")
        print("- Не установлен/не подходит NVIDIA драйвер (проверь nvidia-smi)")
        print("- Установлены CPU wheels PyTorch вместо CUDA wheels")
        return 2

    # Информация о GPU
    try:
        n = torch.cuda.device_count()
        print(f"CUDA device count: {n}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name} | {props.total_memory/1024**3:.2f} GB | CC {props.major}.{props.minor}")
    except Exception:
        print("⚠️ Не удалось получить свойства устройства.")
        traceback.print_exc()

    # Тест 1: простая тензорная операция на GPU
    try:
        device = torch.device("cuda:0")
        x = torch.randn(1024, 1024, device=device)
        y = torch.randn(1024, 1024, device=device)
        z = x @ y
        torch.cuda.synchronize()
        print("\n✅ GPU matmul OK. Result:", float(z[0, 0]))
    except Exception:
        print("\n❌ Ошибка при GPU matmul.")
        traceback.print_exc()
        return 3

    # Тест 2: небольшой Conv2D (проверка cuDNN)
    try:
        import torch.nn as nn
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device)
        inp = torch.randn(8, 3, 64, 64, device=device)
        out = conv(inp)
        loss = out.mean()
        loss.backward()
        torch.cuda.synchronize()
        print("✅ cuDNN/Conv2d forward+backward OK. out shape:", tuple(out.shape))
    except Exception:
        print("❌ Ошибка при Conv2d/cuDNN тесте.")
        traceback.print_exc()
        return 4

    # Импорт torchvision/torchaudio (если стоят)
    for pkg in ("torchvision", "torchaudio"):
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"✅ {pkg} import OK. version: {ver}")
        except Exception as e:
            print(f"⚠️ {pkg} импорт не удался: {repr(e)}")

    print("\n🎉 Всё выглядит корректно: CUDA доступна и PyTorch выполняет операции на GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())