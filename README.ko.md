# vault-semantic-search

Obsidian 볼트 같은 Markdown 노트 폴더를 뜻으로 찾는 로컬 검색 도구예요. 기억나는 대로 질문하면, CPU에서 도는 작은 다국어 임베딩 모델이 뜻이 가장 가까운 섹션이 있는 노트를 찾아 줘요. 내 컴퓨터 밖으로 나가는 것이 없고, 실행 중에 모델을 내려받지 않아요.

[English README](README.md)

꾸민 노트 세 개로 해 봤어요. `Printer fix.md`와 `Coffee budget.md`는 영어, `산책 기록.md`는 한국어로 썼어요. 한국어로 물으면 영어 노트를, 영어로 물으면 한국어 노트를 찾아요(실제 출력에서 순위 줄만 남김).

```text
$ python -I -B vault_search.py search demo-vault "프린터 종이 걸림"
1. [[Printer fix]] (similarity 0.83)
2. [[산책 기록]] (similarity 0.77)
3. [[Coffee budget]] (similarity 0.69)

$ python -I -B vault_search.py search demo-vault "walking in the park"
1. [[산책 기록]] (similarity 0.78)
2. [[Printer fix]] (similarity 0.76)
3. [[Coffee budget]] (similarity 0.70)
```

점수는 한 번의 검색 안에서 순서를 정할 뿐이고, 얼마나 확실한지를 뜻하지 않아요.

## 문제

낱말 검색(grep, Obsidian 검색창)은 노트에 쓴 낱말을 기억할 때만 노트를 찾아요. 몇 달 뒤에는 무슨 내용이었는지는 기억해도 어떻게 썼는지는 잊어요. 「해 뜨기 전에 물 주기가 얼마나 돌지」라고 물어도 노트에는 「점적 호스가 매일 아침 20분 돈다」라고 적혀 있어요. 노트로 질문에 답하는 에이전트도 같아서, grep이 몇 번 빗나가면 「기록이 없다」고 답하기 쉬워요.

vault-semantic-search는 grep 옆에 뜻 검색을 더해요.

1. 노트를 제목(heading) 섹션으로 나누고, 각 섹션을 모델 입력 512토큰에 맞는 겹치는 창으로 나눠요.
2. 창마다 `intfloat/multilingual-e5-small`(ONNX, CPU)로 한 번 임베딩해서 볼트 밖 캐시에 둬요. 다음 실행부터는 새 글이나 바뀐 글만 임베딩해요.
3. 질문도 같은 방식으로 임베딩해서 모든 창과 비교해요. 노트마다 가장 가까운 창 하나가 그 노트의 순위를 정해요.
4. 질문 낱말이 노트 제목, 별칭, 지정한 front matter 필드와 정확히 같으면 앞에 올려요(자리의 절반까지). 기억나는 이름이 느슨한 뜻 일치 뒤로 밀리지 않게요.
5. `warm` 명령으로 밤마다 캐시를 채워 두면 낮에 검색할 때 임베딩을 기다리지 않아요.

## 하는 일과 하지 않는 일

- 볼트 폴더(또는 `--in` 폴더) 아래의 `*.md` 파일만 읽고, 노트를 쓰지 않아요. 쓰는 파일은 자기 임베딩 캐시와, `recall`에 결과 파일을 달라고 했을 때의 그 파일뿐이에요(볼트 안에는 노트가 아닌 새 파일로만 써요).
- 볼트 안의 링크, junction, 그 밖의 reparse point는 따라가지 않아요. 클라우드 자리표시 파일(OneDrive 「필요할 때 파일 다운로드」)은 그대로 읽어요.
- 폴더를 아예 빼거나(`exclude_dirs`), 검색은 하되 글 조각과 섹션 제목을 숨길 수 있어요(`private_dirs`, `hide_snippets_when`). 뺀 폴더는 그 안의 파일을 열기 전에 건너뛰어요.
- 결과는 후보예요. 결과 개수는 `--top` 상한이지 전체 개수가 아니고, 결과가 적거나 없어도 볼트에 그 내용이 없다는 근거가 아니에요. 모든 출력이 이 점을 적어요.
- 출력 문구는 영어예요. 질문과 노트는 모델이 지원하는 어느 언어든 되고, 영어와 한국어로 시험했어요.
- 대화나 질의응답 도구가 아니고 LLM을 부르지 않아요. 전문 색인도 아니라서, 정확한 글자열은 grep을 쓰세요.
- 볼트를 지켜보지 않아요. 새 노트는 다음 검색(그때 임베딩해요)이나 다음 `warm`에서 찾아요.

## 필요한 것

- Python 3.10 이상을 목표로 해요. 시험은 Windows 11(ARM64)의 Python 3.14에서만 했어요.
- `numpy`, `onnxruntime`, `tokenizers`, `PyYAML`(front matter), `markdown-it-py`(코드 블록). `requirements.txt`를 보세요. 시험한 판은 그 파일에 적은 것뿐이에요. PyTorch는 필요 없어요.
- [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)의 ONNX 판(약 470MB). 모델 라이선스는 모델 카드를 보세요.

Windows, macOS, Linux에서 돌도록 만들었어요. 운영체제마다 다른 호출은 캐시 잠금 하나로, Windows에서는 `msvcrt.locking`, 그 밖에서는 `fcntl.flock`을 써요(둘 다 표준 라이브러리). macOS, Linux에서는 돌려 보지 않았어요.

## 설치

```sh
git clone https://github.com/gyulbit-tools/vault-semantic-search
cd vault-semantic-search
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell (cmd는 .venv\Scripts\activate.bat)
# source .venv/bin/activate         # macOS, Linux
python -m pip install -r requirements.txt
```

아래의 `python` 명령은 모두 이 `.venv`의 Python이에요. 활성화한 환경에서 돌리거나, `.venv\Scripts\python`(Windows), `.venv/bin/python`(macOS, Linux)을 바로 부르세요.

모델 파일은 한 번 받아요(`hf` 명령은 `huggingface_hub`에 들어 있어요. 옛 판은 `huggingface-cli download`예요).

```sh
python -m pip install -U huggingface_hub
hf download intfloat/multilingual-e5-small onnx/model.onnx onnx/tokenizer.json --local-dir models/multilingual-e5-small
```

`model.onnx`와 `tokenizer.json`이 든 폴더(여기서는 `models/multilingual-e5-small/onnx`)가 `model_dir`이에요. 도구는 두 파일이 있는지만 보고 해시는 확인하지 않아요. 모델 판을 고정하려면 받을 때 `--revision`을 주세요. 이 초안은 리비전 `614241f6`의 파일로 시험했어요.

## 설정

JSON 파일이에요. 찾는 순서는 `--config 파일`, 환경 변수 `VAULT_SEARCH_CONFIG`, 볼트 루트의 `.vault-search.json`, 그다음 기본값이에요. `VAULT_SEARCH_MODEL_DIR`와 `VAULT_SEARCH_CACHE_DIR`는 어디서 읽은 `model_dir`, `cache_dir`보다 앞서요. 검색에 `model_dir`이 필요한 것 말고는 모든 키가 선택이에요. 모르는 키나 잘못된 형식이면 `CONFIG_INVALID`와 그 키 이름을 내고 멈춰요.

`.vault-search.json`은 볼트로 준 폴더에서만 찾아요. 볼트 일부만 찾으려면 볼트 루트(`.vault-search.json`이 있는 폴더)를 주고 `--in 폴더`를 함께 쓰세요. 하위 폴더 자체를 볼트로 주려면 원래 설정을 `--config`나 `VAULT_SEARCH_CONFIG`로 밝혀 주세요. 아니면 루트 설정 파일의 폴더 규칙이 적용되지 않아요.

볼트 안의 설정 파일은 이름이 하나인(하드 링크가 아닌) 보통 파일이고 그 경로의 어느 폴더도 링크나 junction이 아닐 때만 열어요. 아니면 파일을 읽을 수 없을 때처럼 `CONFIG_UNREADABLE`과 고정 이유를 내고 멈춰요. 파일이 볼트 안인지는 준 경로와 함께 실제 경로로도 정해서, Windows 짧은(8.3) 이름, 대소문자가 다른 철자도 같은 규칙을 따르고(macOS 기본 볼륨처럼 대소문자를 가리지 않는 볼륨에서는 대소문자가 다른 철자의 폴더도 디스크에서 같은 폴더면 볼트 폴더로 알아봐요), 링크나 junction을 거쳐 볼트 안에 닿는 경로(볼트 밖의 링크도, 아직 없는 파일을 가리키는 링크도)는 같은 식으로 거절해요. 기본값을 쓰는 것은 `.vault-search.json`이 없을 때뿐이에요. `--config` 값이 비어 있으면 그 파일로 넘어가지 않고 `CONFIG_UNREADABLE`(`detail=empty configuration path`)로 멈춰요. 볼트 루트 파일도 볼트의 일부라서, 남이 준 볼트는 볼트 밖에 둔 내 `--config` 파일로 돌리세요. 그러면 그 파일을 쓰지 않아요.

```json
{
  "model_dir": "C:/models/multilingual-e5-small/onnx",
  "model_name": "intfloat/multilingual-e5-small",
  "exclude_dirs": ["Archive", "Templates", "/Attachments", "Backups/*"],
  "private_dirs": ["Private"],
  "hide_snippets_when": [
    { "field": "private", "values": ["true"] },
    { "field": "sensitivity", "allow": ["low", "medium"] }
  ],
  "term_fields": ["aliases", "alias", "keywords"],
  "pin_field": "featured",
  "label_fields": ["type", "updated"],
  "require_scope_for_paths": false
}
```

| 키 | 기본값 | 뜻 |
|---|---|---|
| `model_dir` | 없음 | `model.onnx`와 `tokenizer.json`이 든 폴더. |
| `model_name` | `intfloat/multilingual-e5-small` | 캐시에 적는 이름. 모델 파일을 바꾸면 이 이름도 바꾸세요. 아니면 옛 벡터를 다시 써요. |
| `query_prefix`, `passage_prefix` | `"query: "`, `"passage: "` | 질문과 노트 창 앞에 붙이는 글. E5의 접두어예요. 접두어가 없는 모델이면 `""`. |
| `cache_dir` | `%LOCALAPPDATA%\vault-semantic-search`, 없으면 `$XDG_CACHE_HOME/vault-semantic-search`, 없으면 `~/.cache/vault-semantic-search` | 캐시 위치. 볼트마다 하위 폴더(`search.npz`와 잠금 파일 `search.npz.lock`)를 따로 둬요. 동기화되는 볼트 밖에 두세요. |
| `skip_dot_dirs` | `true` | 이름이 `.`으로 시작하는 폴더와 파일(`.obsidian`, `.git` 등)을 건너뛰어요. |
| `skip_dir_names` | `[".git", ".trash", "node_modules"]` | 어느 깊이에서든 건너뛰는 폴더 이름(그 이름의 노트 파일은 건너뛰지 않아요). |
| `exclude_dirs` | `[]` | 읽지 않는 폴더 패턴(아래). |
| `private_dirs` | `[]` | 검색과 목록에는 넣되 글 조각과 섹션 제목 없이 보여 주고, front matter 낱말도 쓰지 않는 폴더 패턴. |
| `hide_snippets_when` | `[]` | `private_dirs`와 같은 효과를 내는 front matter 규칙. `values`: 필드 값이 이 중 하나면 숨겨요. `allow`: 필드 값이 이 중 하나가 아니면 숨겨요(필드가 없는 페이지도 숨겨요). 노트 값과 규칙 값은 둘 다 같은 꼴로 맞춰 비교해요. 앞뒤 공백을 떼고, 여러 줄 값은 공백으로 이어 한 줄로 만들고, 유니코드 NFC 정규화와 대소문자 접기를 해요. |
| `term_fields` | `["aliases", "alias", "keywords"]` | 노트 제목 말고도 정확 낱말로 셀 front matter 필드. |
| `pin_field` | 없음 | 이 front matter 필드가 `true`인 페이지를 정확 낱말 일치의 맨 앞에 둬요. |
| `label_fields` | `[]` | 결과 줄 뒤에 보여 줄 front matter 필드(유형, 날짜 등). |
| `require_scope_for_paths` | `false` | `true`면 `--in` 없는 검색은 JSON 요약만 내고, `--in` 없는 `--output paths`, `details`는 거절해요(`SEARCH_SCOPE_REQUIRED`). 에이전트가 노트 경로를 보기 전에 폴더부터 좁히게 할 때 써요. |

폴더 패턴(`exclude_dirs`, `private_dirs`)은 `/`로 쓴 볼트 기준 폴더 경로예요. 각 부분에 `*`(아무 글자 여럿), `?`(글자 하나)를 쓸 수 있고, `[`, `]`를 비롯한 나머지 글자는 그 글자 그대로 맞춰요. 그래서 `[Old] Notes`는 그 이름의 폴더에 맞아요. 패턴은 어느 깊이에서든 맞아요(`Archive`는 `Projects/Archive`에도 맞아요). 볼트 루트에서만 맞추려면 `/`로 시작하세요. 패턴은 폴더하고만 비교하고 노트 파일 이름과는 비교하지 않아요. `Backups/*`는 `Backups`의 하위 폴더만 빼거나 비공개로 하고 그 바로 아래 노트는 그대로 둬요. 볼트 안의 링크나 junction 가운데 폴더로 이어지고 이름이 `exclude_dirs` 패턴이나 `skip_dir_names` 이름에 맞는 것은 뺀 폴더처럼 건너뛰고, 따라가지도 읽기 문제로 세지도 않아요. 파일로 이어지는 링크(대상을 읽을 수 없는 링크도)는 이름과 상관없이 읽기 문제예요. `/`로 시작하지 않는 `private_dirs` 패턴은 절대 경로 전체로 봐서, 볼트 인자를 비공개 폴더 자체로 줘도 비공개로 남아요. 이름은 유니코드 NFC 정규화와 대소문자 접기 뒤에 비교해요.

## 사용법

### search

```sh
python -I -B vault_search.py search <볼트> "해 뜨기 전에 물 주기가 얼마나 돌지"
python -I -B vault_search.py search <볼트> "질문 하나" "질문 둘" --in Projects --in "Journal/2026" --top 10
```

질문 여러 개를 한 명령에 넣으면 폴더 읽기와 모델 불러오기를 한 번만 해요. `--in`은 볼트 기준 폴더 경로예요(절대 경로는 실제 철자로 볼트 안이어야 하고, 아니면 `SEARCH_SCOPE_INVALID`로 멈춰요). 그 경로에 있는 폴더를 먼저 써요. 그 경로에 없는 폴더 이름 하나만 주면 볼트 어디든 그 이름의 폴더 하나에 맞추고(출력에 `scope_alias_resolved:`가 나와요), 그런 폴더가 둘이면 멈춰요(`SEARCH_SCOPE_AMBIGUOUS`). 이름을 찾는 동안 다 읽지 못한 폴더가 있으면(빼지 않은 링크나 junction, 목록을 읽을 수 없는 폴더, 올바른 유니코드가 아닌 폴더 이름) 그 안에 같은 이름의 폴더가 더 있을 수 있어서 `SEARCH_SCOPE_UNRESOLVED`로 멈춰요. 그 폴더를 빼거나 전체 경로를 주세요. 이름만으로는 `private_dirs` 아래 폴더에 맞추지 않아요. 전체 경로를 주세요.

출력 방식(`--output`):

| 방식 | 보여 주는 것 |
|---|---|
| `details`(기본) | 링크, 섹션 제목, 200자 글 조각. |
| `paths` | 링크와 유사도 점수, `label_fields` 값. 섹션 제목과 글 조각은 없어요(숨긴 페이지도 라벨은 보여요. 다만 `hide_snippets_when` 규칙으로 확인할 수 없는 페이지는 빼요. 한계 절 참고). `details`에서 `--no-snippet`을 줘도 같아요. |
| `summary` | 질문마다 개수만 든 JSON 한 줄, 경로 없음. |
| `auto` | `details`. `require_scope_for_paths`가 켜져 있으면 범위 없는 검색은 `summary`. |

출력 예: `tests/support.py`의 합성 시험 볼트에 `exclude_dirs: ["Archive"]`, `private_dirs: ["Private"]`, `label_fields: ["type"]`로 `intfloat/multilingual-e5-small` ONNX 모델을 돌린 실제 출력이에요.

```sh
python -I -B vault_search.py search <볼트> "how long does the watering run before sunrise" --top 3
```

```text
query: how long does the watering run before sunrise
SEARCH_COMPLETE: coverage_complete=true, candidates=3, chunks=8, read_problems=0
Candidates are exact-term matches plus the top semantic matches. The count is a limit, not a total, and is not evidence of absence.
1. [[Projects/Garden irrigation]] › Timer (similarity 0.89, type: project)
   The drip line runs for twenty minutes every morning before sunrise.
2. [[Notes/Shown by field]] › Plan (similarity 0.84)
   A watering can for the balcony plants.
3. [[Notes/Hidden by field]] › Plan (similarity 0.84)
   The drip watering schedule for the greenhouse.
```

`Private` 아래 페이지가 목록에 들면 글 조각 자리에 `(snippet hidden: private folder or front matter rule)`이 나오고 섹션 제목은 빠져요.

`SEARCH_PARTIAL`(`coverage_complete=false`)은 일부 파일이나 폴더를 읽지 못했다는 뜻이에요(읽을 수 없는 파일, 볼트 안의 링크, 이름이 둘인 노트 파일, UTF-8이나 UTF-16이 아닌 글). 결과는 나오지만 범위 전체를 본 것은 아니에요.

### warm

```sh
python -I -B vault_search.py warm <볼트>
```

볼트 전체의 섹션을 캐시에 임베딩하고 더 쓰지 않는 벡터를 지운 뒤 JSON 한 줄(`WARM_COMPLETE` 또는 `WARM_PARTIAL`과 개수)을 내요. 진행 상황은 stderr로 나가요. 벡터를 지우는 것은 볼트 전체를 빠짐없이 읽었을 때뿐이에요. `warm`이나 범위 없는 검색이 읽기 문제 없이 끝났을 때이고, 노트나 섹션이 하나도 없을 때도 지워요(이때는 모델이 필요 없고, 캐시가 없던 볼트에는 새로 만들지 않아요). 범위를 준 검색이나 읽기 문제가 있는 읽기는 읽지 못한 노트의 벡터를 남겨요. `warm`이 캐시를 저장하거나 정리하지 못하거나 캐시 폴더 위치를 정할 수 없으면(홈 폴더를 찾지 못할 때) 저장할 것이 없어도 `SEARCH_CACHE_WRITE_FAILED`와 종료 코드 2로 멈춰요. 검색은 캐시를 저장하지 못하거나 캐시 폴더 위치를 정할 수 없어도 stderr에 경고를 내고 메모리의 벡터로 답해요.

첫 실행은 전부 임베딩하므로 큰 볼트에서는 오래 걸릴 수 있어요(수십 분). 입력 5,000개마다 중간 저장해서, 끊겨도 다음 실행이 이어서 해요. 밤마다 돌려 두면 검색할 때 캐시가 데워져 있어요.

Windows 작업 스케줄러(PowerShell, `pythonw.exe`는 콘솔 창 없이 돌아요):

```powershell
$action = New-ScheduledTaskAction -Execute "C:\tools\vault-semantic-search\.venv\Scripts\pythonw.exe" `
  -Argument '-I -B -X utf8 "C:\tools\vault-semantic-search\vault_search.py" warm "C:\Notes\MyVault" --config "C:\tools\vault-semantic-search\config.json"'
$trigger = New-ScheduledTaskTrigger -Daily -At 3:30am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName "Vault Search Warm" -Action $action -Trigger $trigger -Settings $settings
```

`-StartWhenAvailable`은 놓친 실행을 컴퓨터가 켜진 뒤에 돌려요. `pythonw.exe`는 출력을 버리므로, 작업의 「마지막 실행 결과」가 종료 코드예요(0 끝, 2 막힘).

cron(macOS, Linux). 줄을 넣기 전에 로그 폴더를 한 번 만들어 두세요. 없으면 셸이 로그를 열지 못해 작업이 돌지 않아요.

```sh
mkdir -p "$HOME/.cache/vault-semantic-search"
```

```cron
30 3 * * * /opt/vault-semantic-search/.venv/bin/python -I -B /opt/vault-semantic-search/vault_search.py warm "$HOME/Notes/MyVault" >> "$HOME/.cache/vault-semantic-search/warm.log" 2>&1
```

### recall

내 볼트에서 기대한 노트를 얼마나 찾는지 재요.

```sh
python -I -B vault_search.py recall <볼트> --questions my-questions.json [--json-out result.json] [--top 10]
```

질문 파일 형식(합성 예는 `examples/recall-questions.example.json`):

```json
{
  "scopes": ["Projects", "Journal", "Recipes"],
  "questions": [
    { "id": "q01", "query": "how long does the watering run before the sun comes up",
      "gold": ["Projects/Garden irrigation.md"] },
    { "id": "q02", "query": "stopping power got weak on my bike",
      "gold": ["Projects/Bicycle repair.md", "Journal/Bike shop visit.md"] }
  ]
}
```

`scopes`는 선택이에요(없으면 볼트 전체). `gold`는 볼트 기준 노트 경로 하나 이상이고, 정답 노트 하나라도 상위 k개 안에 있으면 k에서 맞힌 것으로 세요. 질문은 몇 달 뒤에 물을 말투로 쓰고 노트의 낱말을 그대로 쓰지 마세요. 그러면 grep을 재는 셈이에요.

정답 경로는 폴더 사이를 `/`(또는 `\`)로 쓰고, `.md`는 빼도 돼요. `/`나 드라이브로 시작하거나 `.`, `..` 부분이 들거나, Windows에서 `:`가 들면 막혀요(`RECALL_GOLD_INVALID`). 정답 경로는 철자가 꼭 같은 노트를 가리키고, 그런 노트가 없으면 유니코드 NFC 정규화 뒤에 같은 노트 하나를 가리켜요. 그런 노트가 둘이면 어느 쪽도 채점하지 않고 멈춰요(`RECALL_GOLD_AMBIGUOUS`). `.md`를 쓴 정답 경로는 파일 이름 전체와 비교해서 `Note.md`와 `Note.MD`는 다른 노트예요. `.md`를 뺀 정답 경로는 확장자를 뺀 경로와 비교해서, `Note`는 둘 다를 가리키므로 멈춰요. 결과 파일의 `gold_ranks`는 정답 노트를 그 노트의 확장자 철자로 적어요.

`--json-out`을 주면 결과 파일을 먼저 쓰고 그다음 숫자를 내요. 파일을 쓰지 못하면 `RECALL_BLOCKED code=RECALL_OUTPUT_FAILED`만 내고 숫자는 내지 않아요. `--json-out` 값이 비어 있을 때와, 질문 파일의 글을 UTF-8로 쓸 수 없을 때(id, 질문, 정답 경로, 범위에 든 짝 없는 서로게이트)도 모델을 불러오기 전에 같은 식으로 막아요. 질문 파일과 결과 파일이 볼트 안에 있으면 볼트 안 설정 파일과 같은 규칙을 따라요. 볼트 안의 결과 파일은 새 파일이어야 하고 `.md`로 끝나면 안 돼요(`detail=existing file or note inside the vault`). 실행하는 동안 같은 이름의 파일이 생겼으면 만들지 않아요. 볼트 밖에서는 있던 파일을 바꿔 써요.

보고 형식(예시):

```text
HIT q01 rank=1 query: how long does the watering run before the sun comes up
HIT q02 rank=4 query: stopping power got weak on my bike
     gold Projects/Bicycle repair.md -> 4
     gold Journal/Bike shop visit.md -> not in top
RECALL_COMPLETE questions=2 top1=1/2 (50.0%) top5=2/2 (100.0%) top10=2/2 (100.0%)
```

보고에는 질문 id, 질문, 정답 노트와 그 순위만 나오고 다른 결과 경로는 나오지 않아요. 분모가 제대로 서지 않는 질문 세트는 0%로 내지 않고 모델을 불러오기 전에 종료 코드 2로 멈춰요. 빈 세트, 정답 없는 질문, 검색이 읽지 않을 정답 노트(없음, `scopes` 밖, 제외, 빈 파일), 다 읽지 못한 폴더가 그래요.

## 얼마나 찾는지

2026-11-20에 실제 개인 볼트 하나에서, 이 초안을 뽑아 온 비공개 도구(같은 조각 나누기, 모델, 순위 규칙)로 쟀어요.

| | 맞힘 | 재현율 |
|---|---|---|
| top 1 | 14 / 30 | 46.7% |
| top 5 | 21 / 30 | 70.0% |
| top 10 | 25 / 30 | 83.3% |

- 볼트 주인이 쓴 질문 30개예요. 각 질문은 정답 노트가 쓰지 않은 낱말로 그 노트의 뜻을 물어요(낱말 찾기가 아니라 뜻 질문). 검색은 `--in`으로 정답 노트가 있는 폴더로 좁혔어요.
- 모델: `intfloat/multilingual-e5-small`, ONNX, CPU. 노트와 질문은 주로 한국어이고 영어 용어가 섞여 있어요.
- 질문과 정답 노트는 개인 자료라 공개하지 않아요. 이 숫자는 기대치일 뿐 다시 재 볼 수 있는 벤치마크가 아니에요. 내 볼트에서 `recall`로 재세요.

이어서 이 저장소의 코드를 같은 볼트, 같은 질문 세트에, 비공개 도구의 폴더 규칙을 옮긴 설정으로 돌렸어요. 30개 질문 모두 순위가 같았어요(맞힘 14, 21, 25).

### 단순 grep 기준선과 비교

2026-11-23에 같은 볼트, 같은 질문 30개로 쟀어요. 독자가 두 방법이 얼마나 겹치는지 물어서 잰 거예요. grep 기준선은 돌리기 전에 정해 뒀어요. 질문을 낱말로 나누고, 흔한 한국어 조사를 떼고, 2글자 이상인 낱말만 남겨요. 그다음 질문 낱말이 서로 다른 것으로 몇 개 들었는지로 노트 순위를 매기고, 같으면 전체 일치 수로 가려요.

| | top 1 | top 5 | top 10 |
|---|---|---|---|
| 뜻 검색 | 14 / 30 | 21 / 30 | 25 / 30 |
| grep 기준선 | 3 / 30 | 8 / 30 | 16 / 30 |

top 10 기준으로 질문별 겹침은 이래요. 둘 다 정답 노트를 찾은 질문이 13개, 뜻 검색만 찾은 질문이 12개, grep만 찾은 질문이 3개, 둘 다 못 찾은 질문이 2개예요.

이 비교는 grep에 불리해요. 질문을 일부러 노트의 낱말 없이 썼기 때문이에요. 키워드 벤치마크가 아니라, 뜻은 기억나는데 낱말이 기억나지 않을 때 grep이 놓치는 것으로 읽으세요. grep만 찾은 질문이 3개 있어서, 이 도구는 grep을 대신하지 않고 grep 곁에 둬요.

속도는 Windows 11 ARM64 노트북, Python 3.14, onnxruntime 1.32(CPU), 데운 캐시에서 쟀어요. 비공개 도구로 질문 하나에 약 1.7초, 이 코드로 recall 질문 30개를 한 명령에 약 18초(폴더 읽기와 모델 불러오기 한 번)였어요. 찬 캐시는 새 글을 임베딩하는 시간만큼 더 걸려요.

## 한계

- **작은 모델 하나.** e5-small은 빠르고 다국어를 지원하지만 가장 센 모델은 아니에요. 풀링은 E5 sentence-transformers 판과 같은 평균 풀링 + L2 정규화예요. CLS 풀링이나 다른 입력 이름을 쓰는 모델은 코드를 고치지 않으면 안 돼요. 시험한 모델은 `intfloat/multilingual-e5-small`뿐이에요.
- **긴 섹션**은 최대 500자(겹침 100자) 창으로 나누고, 제목 머리와 함께 512토큰에 맞게 잘라요. 노트 순위는 가장 가까운 창 하나로 정해요.
- **정확 낱말**은 유니코드 NFC와 대소문자 접기 뒤의 온전한 낱말이에요. 어간 처리나 부분 문자열은 없어요. 검색한 페이지의 5분의 1 이상에 나오는 낱말은 너무 흔해서 무시해요.
- **front matter**는 평범한 YAML만 읽어요. 스칼라, 목록, 매핑이고, 값은 쓴 글 그대로 둬요(`true`나 `2026-01-02`도 글이에요). 태그(`!!int 10`, `!!str x`, `!custom x`), 앵커(`&a`), 별칭(`*a`), 병합 키(따옴표 없는 `<<`)가 어디든 하나라도 있으면 front matter 전체를 읽지 못한 것으로 봐요. PyYAML의 C 확장이 깔려 있어도 순수 Python 파서로 읽어요. 두 파서가 받아들이는 글이 달라서, 어느 컴퓨터에서든 같은 노트를 같게 읽게 하려는 거예요. 첫 줄 `---`로 시작해서 줄 맨 앞에 `---`만 있는(뒤에 공백이나 탭은 돼요) 다음 줄에서 끝나요. `...`로는 끝나지 않아요. 줄은 줄바꿈(`\n`, `\r\n` 안의 것도)에서만 끝나요. 홀로 있는 `\r`, U+0085, U+2028, U+2029에서는 줄이 끝나지 않아요. 줄바꿈과 탭 말고 제어 문자나 줄, 문단 구분 문자(홀로 있는 `\r`, U+0085, U+2028, U+2029, U+000B, U+000C 등)가 든 front matter와 공백 문자만 든 front matter도 읽지 못한 것으로 봐요. YAML은 이 도구가 줄을 끝내지 않는 자리에서 줄을 끝내기 때문이에요. 빈 front matter(`---` 바로 다음 줄이 `---`)는 필드가 없는 것으로 읽어요. 맨 위 단계의 글 값과 글 목록을 쓰고, `>`나 `|` 블록은 한 줄로 합쳐요. 중첩 매핑과 두 번 쓴 키는 읽지 않아요. `hide_snippets_when` 규칙이 없으면 front matter를 읽지 못한 페이지도 검색하고 보여 주지만, 그 front matter는 하나도 쓰지 않아요(라벨, 정확 낱말, `pin_field` 없음).
- **규칙으로 확인할 수 없는 페이지.** `hide_snippets_when` 규칙이 있을 때, 규칙을 확인할 수 없는 페이지는 숨기고 라벨 없이 링크만 보여요. 파일이 UTF-8 글이 아니거나(UTF-16, NUL이 든 글도), 첫 줄이 `---`인데 닫는 `---` 줄이 없거나, 첫 줄이 다른 공백 문자를 떼어야만 `---`이거나(`---` 뒤에 U+2028, U+2029, U+0085, 줄바꿈 없는 공백이 붙었거나, 공백이나 탭 뒤에 `---`가 있을 때. 다른 프로그램은 그 노트에 front matter가 없다고 볼 수 있어요), front matter를 읽지 못했거나(올바른 YAML이 아니거나, 태그, 앵커, 별칭, 병합 키를 썼거나, 제어 문자나 줄 구분 문자가 들었거나, 공백 문자만 든 것), 규칙 필드가 중첩 값이거나 두 번 쓰였을 때예요. 그 front matter는 정확 낱말과 `pin_field`에도 쓰지 않아요. 검색은 다른 페이지로 이어 가요. 규칙이 없으면 그런 페이지도 검색하고 보여 줘요. 다만 front matter를 읽지 못했거나(위 참고) 첫 줄이 다른 공백 문자를 떼어야만 `---`이면 그 front matter는 쓰지 않아요(라벨, 정확 낱말, `pin_field` 없음). 나머지는 전처럼 읽어요.
- **숨김은 표시만 막아요.** 숨긴 페이지도 캐시에는 임베딩되고, 링크는 보여요. 라벨도 보이지만 위 페이지는 빼요. 캐시 파일에는 노트 글이 아니라 벡터와 SHA-1 열쇠가 들어 있지만, 노트에서 나온 자료로 다루세요.
- **하드 링크.** 이름이 둘 이상인 노트 파일은 열기 전에 거절하고 읽기 문제로 세요(그래서 `recall`은 멈춰요). 이름 하나가 비공개 폴더나 뺀 폴더 밖에 있을 수 있는데, 어느 이름이 원래 것인지 도구가 가릴 수 없어서예요. 파일 시스템이 링크 수를 알려 줄 때만 돼요. 알려 주지 않으면 다른 파일처럼 읽어요. 그 노트를 찾으려면 이름을 하나만 남기세요.
- **코드 블록**(CommonMark 규칙의 울타리 블록과 들여쓴 블록, 목록과 인용 안의 것도)은 검색하지 않고, 인라인 코드는 백틱만 벗겨 검색해요. 임베드(`![[...]]`)와 이미지 링크는 빼고 링크 글자는 남겨요.
- **캐시 열쇠**는 모델 이름과 노트 창의 입력 글 그대로예요. 조각 나누기나 `passage_prefix`를 바꾸면 최종 입력 글이 바뀐 창만 다시 임베딩해요(`passage_prefix`를 바꾸면 모든 창이 바뀌어요). 질문만의 캐시 항목은 없고 노트 창이 하나 이상일 때 질문은 실행마다 임베딩해요. 다만 질문의 최종 입력(`query_prefix`와 질문)이 노트 창의 입력과 같으면 그 창의 벡터를 써요. 조각이 0개면 정확한 낱말 일치 결과만 내고 모델을 불러오지 않아요. `query_prefix`를 바꿔도 다시 임베딩하는 노트 창은 없어요. 캐시 파일은 형식 번호가 정수 2이고, 열쇠마다 유한한 단위 길이 벡터가 하나씩 있고, 모델 이름, 열쇠, 벡터로 만든 SHA-256 요약값이 맞을 때만 써요. 아니면(손상된 파일도) 다시 만들어요. 모델이 새로 계산한 벡터도 노트 창의 것이든 질문의 것이든 같은 규칙을 따라요. 유한하지 않거나 단위 길이가 아닌 벡터(질문 벡터는 창 벡터와 폭이 다른 것도)는 저장하거나 순위에 쓰기 전에 `SEARCH_FAILED`로 멈춰요(그 앞에 중간 저장한 것은 남아요). 같은 캐시를 저장하는 실행들은 잠금 파일로 차례를 지키고, 저장할 때마다 그사이 다른 실행이 저장한 벡터를 남겨요. 벡터를 지우는 것은 볼트 전체를 빠짐없이 읽은 실행뿐이에요. 30초 안에 잠금을 얻지 못하면 저장 실패로 다뤄요. 올바른 캐시라도 벡터 폭이 모델과 다르면(같은 `model_name`에 다른 모델 파일) 캐시를 바꾸기 전에 `SEARCH_FAILED`로 멈춰요. 이때는 `model_name`을 바꾸세요. 볼트마다 캐시 폴더 이름은 파일 시스템이 적는 볼트 경로 철자에서 만들어요(macOS, Linux에서는 폴더마다 그 부모 폴더 목록에 적힌 이름). 그래서 macOS 기본 볼륨처럼 대소문자를 가리지 않는 볼륨에서 같은 볼트를 대소문자가 다른 철자로 줘도 같은 캐시를 써요. 다만 부모 폴더의 목록을 읽을 수 없으면 그 바로 아래 경로 성분은 준 철자 그대로라, 그 성분의 대소문자를 다르게 주면 캐시가 따로 생겨요.
- **색인 구조 없음.** 검색마다 질문을 메모리의 모든 벡터와 비교해요. 섹션 수만 개는 괜찮지만 수백만 개에는 맞지 않아요.
- **로컬 볼트 경로만.** UNC 경로(`\\server\share\...`, `\\?\UNC\...`, `--in` 값으로 줘도)와 연결된 네트워크 드라이브는 거절하고(`SEARCH_ROOT_INVALID`), 볼트 폴더나 그 위쪽 폴더 중 하나가 symlink나 junction인 경로도 거절해요. 비공개 폴더 확인과 범위 확인이 늘 실제 경로를 보게 하려는 거예요. macOS의 `/tmp`(`/private/tmp`로 가는 링크) 같은 시스템 링크도 여기에 들어서, 실제 경로를 주세요.
- **결과 경로**는 볼트 기준 링크(`[[폴더/노트]]`)예요. 검색한 노트 둘의 링크가 같아질 때(`Note.md`와 `Note.MD`)는 그 링크에 확장자를 붙여요(`[[폴더/Note.md]]`, `[[폴더/Note.MD]]`). 그중 어느 것이 결과에 들든 같아요. 그렇게 붙인 링크와 같아지는 노트(그 둘 옆의 `Note.md.md`, 확장자를 뺀 링크가 `[[폴더/Note.md]]`)도 확장자를 붙여요. 링크는 철자 그대로 비교해서 `Note.md`와 `note.md`는 `[[폴더/Note]]`, `[[폴더/note]]`예요. 막힌 검색은 고정 코드(`CONFIG_INVALID`면 키 이름도, 볼트 안 설정 파일을 거절했으면 고정 이유도)가 든 JSON 한 줄만 내고, 경로, 질문, 노트 글은 내지 않아요. 명령줄 인자가 틀려도(범위를 벗어난 `--top`도) 같은 식으로 `SEARCH_ARGS_INVALID`만 내고 인자는 적지 않아요. `VAULT_SEARCH_DEBUG=1`이면 stderr에 traceback을 보여 줘요. 막힌 `recall`은 내 질문 파일의 질문 id와 정답 경로를 적어요. 이름을 UTF-8로 쓸 수 없는 폴더나 노트(깨진 유니코드 이름)는 읽기 문제로 세고 건너뛰어요. `--in`으로 준 폴더도 같고(다른 `--in` 폴더 안에 있거나 여러 `--in` 값이 같은 깨진 폴더에서 막혀도 한 번만 세요), 다른 `--in` 폴더는 그대로 찾으며 볼트 전체로 넓히지 않아요.

## 시험

```sh
python -I -B tests/run_tests.py
```

설치에서 만든 `.venv`의 Python으로 돌리세요. 시험은 임시 폴더에 작은 합성 볼트를 만들고 가짜 인코더를 써서, `numpy`, `PyYAML`, `markdown-it-py`만 있으면 모델 없이 돌아요(내장 ONNX 로더가 아닌 로더로 찾을 때는 `numpy`만 확인해요). `VAULT_SEARCH_MODEL_DIR`에 모델 폴더를 주면 ONNX 확인 시험(`tests/test_onnx_model.py`)도 돌아요. junction 시험, 짧은 이름 시험, 확장 경로 시험은 Windows에서만 돌고, symlink 시험은 계정이 symlink를 만들 수 없으면 건너뛰어요. 마지막 줄은 `RESULT pass=N fail=M skip=K`이고, 하나라도 실패하면 종료 코드 1이에요.

## 라이선스

MIT. [LICENSE](LICENSE)를 보세요. 모델은 이 저장소에 들어 있지 않고 자기 라이선스를 따라요.
