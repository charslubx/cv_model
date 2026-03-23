# Process Change Request – Doc Generator

A Django web application that generates a formatted `.docx` (Word) document
with a **Process Change Request** first page, matching the layout:

```
┌──────────────────────────────────────────┐
│  DOC-2024-001                            │
│     PROCESS CHANGE REQUEST               │
│     Engineering Change Management Form   │
├──────────────────────────────────────────┤
│ INFO                                     │
│  1. White Paper Type  │  <value>         │
│  2. Classification    │  <value>         │
├──────────────────────────────────────────┤
│ INFO                                     │
│  3. Name of Owner         │  <value>     │
│  4. Title of Change       │  <value>     │
│  5. Change Description    │  <value>     │
│  6. Reason for Change     │  <value>     │
├──────────────────────────────────────────┤
│ INFO                                     │
│  7. Process Factors                      │
│  Process Factor │ Present │ Proposed     │
│  …              │  …      │  …           │
└──────────────────────────────────────────┘
Rev 2.0                              Page 1
```

## Quick Start

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Then open <http://127.0.0.1:8000/> in your browser, fill in the form, and
click **Generate Document** to download the `.docx` file.

## API

### `GET /generate/`
Downloads a demo `.docx` with sample data (no body required).

### `POST /generate/`
Body: JSON object with the following fields:

| Field | Description |
|---|---|
| `doc_number` | Document reference number |
| `document_title` | Main title (all-caps recommended) |
| `document_subtitle` | Subtitle below the main title |
| `white_paper_type` | Field 1 value |
| `classification` | Field 2 value |
| `name_of_owner` | Field 3 value |
| `title_of_change` | Field 4 value |
| `change_description` | Field 5 value |
| `reason_for_change` | Field 6 value |
| `process_factors` | Array of `{factor, present_value, proposed_value}` |
| `rev_number` | Footer revision label (e.g. `Rev 2.0`) |

## Project Structure

```
doc_generator_project/
├── doc_generator/          # Django project config
│   ├── settings.py
│   └── urls.py
├── docgen/                 # Application
│   ├── views.py            # Document generation logic
│   ├── urls.py
│   └── templates/
│       └── docgen/
│           └── index.html  # Web form UI
├── manage.py
└── requirements.txt
```
