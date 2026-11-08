import sounddevice as sd

devices = [9, 13, 1]

print("기본 입력 장치:", sd.default.device[0])

for i in devices:
    print(f"\n[{i}번 장치 검사]")

    try:
        d = sd.query_devices(i)
        print("이름:", d["name"])
        print("기본 주파수:", d["default_samplerate"])
        print("입력 채널:", d["max_input_channels"])

        if d["max_input_channels"] < 1:
            print("입력 불가능한 장치")
            continue

        sd.check_input_settings(
            device=i,
            channels=1,
            samplerate=int(d["default_samplerate"])
        )

        print("입력 설정 사용 가능")

    except Exception as e:
        print("오류:", repr(e))