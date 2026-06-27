import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, url_for


app = Flask(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DOGOVOR_FIELDS = {
    'number': 'Номер договора',
    'name': 'Наименование',
    'counterparty': 'Контрагент',
    'subject': 'Предмет договора',
    'amount': 'Сумма',
    'status': 'Статус',
    'responsible': 'Ответственный',
    'notes': 'Примечания',
    'contract_type': 'Тип (основной/ДС)',
    'received_date': 'Дата получения',
    'processing_date': 'Дата оформления',
    'approval_date': 'Дата согласования',
    'signing_date': 'Дата подписания',
    'sent_date': 'Дата направления',
    'archive_date': 'Дата архивации',
    'destroyed_date': 'Дата уничтожения',
}

DOGOVOR_RUSSIAN_ALIASES = {
    'номер': 'number', '№': 'number', 'регистрационный': 'number',
    'наименование': 'name', 'название': 'name', 'договор': 'name',
    'контрагент': 'counterparty', 'контрагент/партнер': 'counterparty',
    'партнер': 'counterparty', 'партнёр': 'counterparty',
    'контрагент/партнёр': 'counterparty', 'организация': 'counterparty',
    'предмет': 'subject', 'содержание': 'subject', 'описание': 'subject',
    'сумма': 'amount', 'цена': 'amount', 'стоимость': 'amount',
    'статус': 'status', 'этап': 'status', 'состояние': 'status',
    'ответственный': 'responsible', 'исполнитель': 'responsible',
    'менеджер': 'responsible', 'куратор': 'responsible',
    'примечание': 'notes', 'комментарий': 'notes', 'заметки': 'notes',
    'тип': 'contract_type', 'вид': 'contract_type',
}

NUMERIC_SUGGEST_FIELDS = ['amount']

DATE_FIELDS = ['received_date', 'processing_date', 'approval_date',
               'signing_date', 'sent_date', 'archive_date', 'destroyed_date']


def to_native(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, (pd.Timestamp,)):
        return val.isoformat()
    if isinstance(val, (pd.Timedelta,)):
        return str(val)
    return val


def analyze_dataframe(df):
    total_rows = int(len(df))
    columns = []
    suggestions = []
    missing_required = []

    for col in df.columns:
        col_str = str(col).strip()
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        null_pct = round(null_count / total_rows * 100, 1) if total_rows else 0
        unique_count = df[col].nunique()
        sample_values = df[col].dropna().head(5).tolist()

        dtype = str(df[col].dtype)
        inferred_type = infer_column_type(df[col], dtype, sample_values)

        mapped_field = detect_field(col_str)
        mapping_confidence = 'high' if mapped_field else None

        columns.append({
            'name': col_str,
            'dtype': dtype,
            'inferred_type': inferred_type,
            'total': to_native(total_rows),
            'non_null': to_native(non_null),
            'null_count': to_native(null_count),
            'null_pct': to_native(null_pct),
            'unique_count': to_native(unique_count),
            'sample_values': [to_native(s) for s in format_samples(sample_values)],
            'mapped_field': mapped_field,
            'mapping_confidence': mapping_confidence,
        })

        if mapped_field:
            suggestions.append({
                'column': col_str,
                'mapped_to': mapped_field,
                'label': DOGOVOR_FIELDS[mapped_field],
            })
        else:
            close = find_close_match(col_str)
            if close:
                suggestions.append({
                    'column': col_str,
                    'mapped_to': None,
                    'suggest_rename': close,
                    'label': DOGOVOR_FIELDS[close],
                })

    missing_required = find_missing_fields(columns)

    return {
        'total_rows': total_rows,
        'total_columns': len(columns),
        'columns': columns,
        'suggestions': suggestions,
        'missing_required': missing_required,
        'has_dates': any(c['inferred_type'] == 'date' for c in columns),
        'has_amounts': any(c['inferred_type'] == 'numeric_amount' for c in columns),
    }


def infer_column_type(series, dtype, samples):
    if dtype == 'object':
        if len(samples) > 0:
            for s in samples:
                if is_date_string(s):
                    return 'date'
        return 'text'
    if 'int' in dtype:
        return 'integer'
    if 'float' in dtype:
        return 'numeric_amount'
    if 'datetime' in dtype or 'datetime64' in dtype:
        return 'date'
    return dtype


def is_date_string(val):
    if not val:
        return False
    val = str(val).strip()
    for fmt in ('%d.%m.%Y', '%d.%m.%y', '%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d'):
        try:
            datetime.strptime(val, fmt)
            return True
        except ValueError:
            pass
    return False


def detect_field(col_name):
    col_lower = str(col_name).lower().strip()
    if col_lower in DOGOVOR_RUSSIAN_ALIASES:
        return DOGOVOR_RUSSIAN_ALIASES[col_lower]
    for alias, field in DOGOVOR_RUSSIAN_ALIASES.items():
        if alias in col_lower or col_lower in alias:
            return field
    return None


def find_close_match(col_name):
    col_lower = str(col_name).lower().strip()
    best = None
    best_score = 0
    for alias, field in DOGOVOR_RUSSIAN_ALIASES.items():
        score = 0
        for word in alias.split('/'):
            if word in col_lower or col_lower in word:
                score += len(word)
        if score > best_score:
            best_score = score
            best = field
    if best_score > 2:
        return best
    return None


def find_missing_fields(columns):
    mapped = [c['mapped_field'] for c in columns if c['mapped_field']]
    names = [str(c['name']).lower() for c in columns]

    required = ['number', 'name', 'counterparty']
    missing = []
    for f in required:
        if f not in mapped and DOGOVOR_FIELDS[f] not in names:
            missing.append({
                'field': f,
                'label': DOGOVOR_FIELDS[f],
                'critical': True,
            })

    important = ['subject', 'amount', 'status', 'responsible']
    for f in important:
        if f not in mapped:
            missing.append({
                'field': f,
                'label': DOGOVOR_FIELDS[f],
                'critical': False,
            })
    return missing


def format_samples(samples):
    return [str(s)[:80] for s in samples]


def generate_suggestions_text(analysis, filename):
    lines = []
    lines.append(f"Анализ файла: {filename}")
    lines.append(f"Строк: {analysis['total_rows']}, колонок: {analysis['total_columns']}")
    lines.append("")

    if analysis['suggestions']:
        lines.append("=== Соответствие полей ===")
        for s in analysis['suggestions']:
            if s['mapped_to']:
                lines.append(f"  {s['column']} → {s['mapped_to']} ({s['label']})")
            else:
                lines.append(f"  {s['column']} → предлагается переименовать в {s['suggest_rename']} ({s['label']})")
        lines.append("")

    if analysis['missing_required']:
        lines.append("=== Отсутствующие поля ===")
        for m in analysis['missing_required']:
            tag = "ОБЯЗАТЕЛЬНО" if m['critical'] else "опционально"
            lines.append(f"  {m['label']} ({m['field']}) — {tag}")
        lines.append("")

    for col in analysis['columns']:
        if col['mapped_field']:
            continue
        lines.append(f"Колонка «{col['name']}» не сопоставлена")
        lines.append(f"  Тип: {col['inferred_type']}, пропусков: {col['null_pct']}%")
        if col['unique_count'] <= 10 and col['null_pct'] < 80:
            lines.append(f"  Примеры: {', '.join(str(s) for s in col['sample_values'][:3])}")
        lines.append("")

    if analysis['has_dates']:
        lines.append("Обнаружены колонки с датами — можно сопоставить с датами жизненного цикла договора.")
    if analysis['has_amounts']:
        lines.append("Обнаружены числовые поля — возможно, суммы договоров.")

    lines.append("")
    lines.append("=== Рекомендации ===")
    if analysis['missing_required']:
        crit = [m for m in analysis['missing_required'] if m['critical']]
        if crit:
            lines.append(f"  Не хватает обязательных полей: {', '.join(m['label'] for m in crit)}.")
            lines.append("  Потребуется доработка программы dogovor или добавление этих данных в файл.")
    lines.append("  Загрузите отчёт JSON для передачи агенту.")

    return '\n'.join(lines)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Файл не выбран'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ('.xlsx', '.xls', '.csv'):
        return jsonify({'error': 'Поддерживаются только .xlsx, .xls, .csv'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(filepath)

    try:
        if ext == '.csv':
            df = pd.read_csv(filepath, encoding_errors='replace')
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        return jsonify({'error': f'Ошибка чтения: {str(e)}'}), 400

    analysis = analyze_dataframe(df)
    suggestions_text = generate_suggestions_text(analysis, f.filename)

    report = {
        'filename': f.filename,
        'analyzed_at': datetime.now().isoformat(),
        'analysis': analysis,
        'suggestions_text': suggestions_text,
    }

    report_path = os.path.join(app.config['UPLOAD_FOLDER'],
                               f'{os.path.splitext(f.filename)[0]}_report.json')
    with open(report_path, 'w', encoding='utf-8') as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2, default=str)

    report['report_url'] = f'/download/{os.path.basename(report_path)}'

    return jsonify(report)


@app.route('/download/<filename>')
def download_report(filename):
    return jsonify({'error': 'Use direct path'}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
