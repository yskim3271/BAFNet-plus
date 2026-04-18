# 📱 원격 안드로이드 빌드 환경 연결 가이드

윈도우(Client)에 연결된 휴대폰을 학교 리눅스 서버(Server)가 인식하도록 **SSH 리버스 터널링**을 설정하는 방법입니다.

## ✅ 0. 사전 필수 점검 (최초 1회)

연결 전, 양쪽 컴퓨터의 **ADB 버전이 반드시 일치(또는 유사)**해야 합니다. (버전 차이가 크면 연결 즉시 끊김)

* **확인 명령어:** `adb --version` (양쪽 모두 확인)
* **권장 버전:** 36.x.x 이상 (최신 버전 권장)
* **리눅스 경로:** `~/platform-tools/adb` 사용 권장 (시스템 기본 `adb`는 구버전일 확률 높음)

---

## 1단계: 윈도우 (Client) 설정

먼저 윈도우 터미널(PowerShell)을 열고, 휴대폰 연결 준비를 합니다.

```powershell
# 1. 기존에 꼬인 ADB 프로세스 초기화 (필수)
adb disconnect
adb kill-server

# 2. 휴대폰을 TCP 모드로 전환 (휴대폰 화면에서 '허용' 팝업 확인)
adb tcpip 5555
# 출력 결과: "restarting in TCP mode port: 5555" 확인

# 3. [핵심] 윈도우 PC의 포트를 휴대폰 포트로 연결 (Bridge)
# 설명: 윈도우(Localhost:5555)로 들어온 신호를 휴대폰(Device:5555)으로 전달
adb forward tcp:5555 tcp:5555

```

---

## 2단계: SSH 터널링 연결 (윈도우)

학교 네트워크 보안과 충돌을 피하기 위해 **15555번 포트**를 사용하고, **VPN 끊김 방지 옵션**을 사용합니다.

* **터미널:** 위에서 작업한 PowerShell 창 사용
* **주의:** 연결 후 이 창은 **절대 끄지 마세요** (최소화 OK).

```powershell
# 문법: ssh -R [서버포트]:[로컬호스트]:[로컬포트] [계정]@[서버IP]
# -o ServerAliveInterval=60 : VPN 환경에서 60초마다 신호를 보내 끊김 방지
ssh -o ServerAliveInterval=60 -R 15555:127.0.0.1:5555 yskim@141.223.24.150

```

---

## 3단계: 리눅스 서버 연결 (Server)

이제 리눅스 서버(Cursor 터미널)에서 터널을 통해 휴대폰과 최종 연결합니다.

```bash
# 1. 최신 ADB 경로로 이동 (설치된 경로에 따라 다름)
cd ~/platform-tools

# 2. 기존 ADB 종료 (충돌 방지)
./adb kill-server

# 3. 연결 시도 (반드시 15555 포트 사용)
./adb connect 127.0.0.1:15555
# 출력 결과: "connected to 127.0.0.1:15555"

```

---

## 4단계: 연결 확인 및 실행

연결이 정상적으로 되었는지 확인합니다.

```bash
# 기기 목록 확인
./adb devices

# [정상 결과 예시]
# List of devices attached
# 127.0.0.1:15555    device

```

👉 **이제 Android Studio나 Gradle로 빌드(Run)하면 내 폰에 앱이 설치됩니다!**

---

## 🛠 트러블 슈팅 (안 될 때 체크리스트)

| 증상 | 원인 및 해결책 |
| --- | --- |
| **`failed to connect`** | 1. 윈도우 SSH 창이 켜져 있는지 확인<br>

<br>2. 윈도우에서 `adb forward tcp:5555 tcp:5555` 했는지 확인 |
| **`connected` 후 바로 끊김** | **ADB 버전 불일치.** 양쪽 버전을 똑같이(36버전) 맞췄는지 다시 확인 |
| **`offline`** | 1. 휴대폰 화면 켜서 "USB 디버깅 허용" 눌렀는지 확인<br>

<br>2. `adb kill-server` 후 재시도 |
| **`Warning: remote port forwarding failed`** | 리눅스 서버에 이미 15555 포트가 사용 중. SSH 창 껐다 켜거나 포트 번호 변경 (예: 15556) |

---

이제 이 문서만 있으면 언제든 다시 연결할 수 있습니다. 개발 파이팅하세요!