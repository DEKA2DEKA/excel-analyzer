import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

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
    'дата': 'received_date', 'дата получения': 'received_date',
    'дата подписания': 'signing_date', 'дата начала': 'received_date',
    'дата окончания': 'archive_date', 'дата архивации': 'archive_date',
}


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


def read_raw_rows(filepath, ext, max_rows=20):
    if ext == '.csv':
        import csv
        rows = []
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(row)
        max_cols = max(len(r) for r in rows) if rows else 0
        padded = [r + [''] * (max_cols - len(r)) for r in rows]
        df_raw = pd.DataFrame(padded)
    else:
        df_raw = pd.read_excel(filepath, header=None, nrows=max_rows)
    return df_raw


def detect_header_row(df_raw, max_scan=15):
    header_candidates = []
    scores = []

    for i in range(min(len(df_raw), max_scan)):
        row = df_raw.iloc[i]
        score = 0
        cell_count = 0
        for val in row:
            s = str(val).strip()
            if not s or s.lower() == 'nan':
                continue
            cell_count += 1
            s_lower = s.lower()

            if s_lower in DOGOVOR_RUSSIAN_ALIASES:
                score += 10
            for alias in DOGOVOR_RUSSIAN_ALIASES:
                if alias in s_lower or s_lower in alias:
                    score += 3
                    break

            if any(c.isalpha() for c in s) and not any(c.isdigit() for c in s):
                score += 1

        if cell_count == 0:
            continue

        total_cols = len(df_raw.columns)
        coverage = cell_count / total_cols if total_cols else 0
        if coverage >= 0.4:
            score += int(coverage * 5)

        header_candidates.append({
            'row_index': i,
            'row_number': i + 1,
            'cells': [str(df_raw.iloc[i][c])[:50] for c in range(total_cols)],
            'score': score,
            'cell_count': cell_count,
        })
        scores.append(score)

    best = sorted(header_candidates, key=lambda x: x['score'], reverse=True)
    if not best:
        return {'header_row': 0, 'header_number': 1,
                'confidence': 'low', 'candidates': [], 'best_score': 0}

    top_score = best[0]['score']
    second_best_score = best[1]['score'] if len(best) > 1 else 0

    if top_score >= 15 and top_score > second_best_score * 2:
        confidence = 'high'
    elif top_score >= 8:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'header_row': best[0]['row_index'],
        'header_number': best[0]['row_number'],
        'confidence': confidence,
        'best_score': top_score,
        'second_best_score': second_best_score,
        'candidates': best[:5],
    }


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
    if 'header_number' in analysis:
        lines.append(f"Шапка таблицы: строка {analysis['header_number']} (уверенность: {analysis['header_confidence']})")
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

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)

    header_row = request.form.get('header_row', type=int, default=None)

    if header_row is None:
        try:
            df_raw = read_raw_rows(filepath, ext, max_rows=20)
        except Exception as e:
            return jsonify({'error': f'Ошибка чтения: {str(e)}'}), 400

        header_info = detect_header_row(df_raw)

        if header_info['confidence'] in ('medium', 'low') and len(header_info['candidates']) > 1:
            preview_rows = []
            for c in header_info['candidates'][:5]:
                preview_rows.append({
                    'row_number': c['row_number'],
                    'cells': c['cells'],
                    'score': c['score'],
                })
            return jsonify({
                'status': 'needs_header_selection',
                'header_info': header_info,
                'preview_rows': preview_rows,
                'message': f'Уверенность: {header_info["confidence"]}. Выберите строку заголовков.',
            })

        selected_row = header_info['header_row']
        header_number = header_info['header_number']
        header_confidence = header_info['confidence']
    else:
        selected_row = max(0, header_row - 1)
        header_number = header_row
        header_confidence = 'manual'

    try:
        if ext == '.csv':
            import csv
            all_rows = []
            with open(filepath, 'r', encoding='utf-8', errors='replace') as csvf:
                reader = csv.reader(csvf)
                for row in reader:
                    all_rows.append(row)
            if not all_rows:
                return jsonify({'error': 'Файл пуст'}), 400
            max_cols = max(len(r) for r in all_rows)
            padded = [r + [''] * (max_cols - len(r)) for r in all_rows]
            headers = padded[selected_row]
            data_rows = padded[selected_row + 1:]
            df = pd.DataFrame(data_rows, columns=headers)
        else:
            df = pd.read_excel(filepath, header=selected_row)
    except Exception as e:
        return jsonify({'error': f'Ошибка чтения: {str(e)}'}), 400

    if df.empty or len(df.columns) == 0:
        return jsonify({'error': 'Таблица пуста или не удалось определить колонки'}), 400

    analysis = analyze_dataframe(df)
    analysis['header_number'] = header_number
    analysis['header_confidence'] = header_confidence

    suggestions_text = generate_suggestions_text(analysis, filename)

    report = {
        'status': 'complete',
        'filename': filename,
        'analyzed_at': datetime.now().isoformat(),
        'analysis': analysis,
        'suggestions_text': suggestions_text,
    }

    report_filename = f'{os.path.splitext(filename)[0]}_report.json'
    report_path = os.path.join(app.config['UPLOAD_FOLDER'], report_filename)
    with open(report_path, 'w', encoding='utf-8') as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2, default=str)

    report['report_url'] = f'/api/download/{report_filename}'

    return jsonify(report)


@app.route('/api/preview-header', methods=['POST'])
def api_preview_header():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Файл не выбран'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ('.xlsx', '.xls', '.csv'):
        return jsonify({'error': 'Поддерживаются только .xlsx, .xls, .csv'}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(filepath)

    try:
        df_raw = read_raw_rows(filepath, ext, max_rows=15)
    except Exception as e:
        return jsonify({'error': f'Ошибка чтения: {str(e)}'}), 400

    header_info = detect_header_row(df_raw)
    preview_rows = []
    for c in header_info['candidates'][:5]:
        preview_rows.append({
            'row_number': c['row_number'],
            'cells': c['cells'],
            'score': c['score'],
        })

    return jsonify({
        'header_info': header_info,
        'preview_rows': preview_rows,
        'filename': filename,
    })


@app.route('/api/download/<filename>')
def download_report(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Файл не найден'}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)


@app.route('/api/generate-task', methods=['POST'])
def api_generate_task():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Нет данных'}), 400

    filename = data.get('filename', 'файл')
    header_row = data.get('header_row', 1)
    columns = data.get('columns', [])

    enabled = [c for c in columns if c.get('included')]
    if not enabled:
        return jsonify({'error': 'Не выбрано ни одной колонки'}), 400

    lines = []
    lines.append(f"Задание на разработку: импорт данных из файла «{filename}»")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Источник: {filename}")
    lines.append(f"Строка заголовков: {header_row}")
    lines.append(f"Количество отобранных колонок: {len(enabled)}")
    lines.append("")

    lines.append("Структура данных:")
    lines.append("-" * 40)
    for c in enabled:
        desc = c.get('description', '').strip() or '—'
        mapped = c.get('mapped_field', '') or '—'
        lines.append(f"  • {c['name']}")
        lines.append(f"    Описание: {desc}")
        if mapped != '—':
            lines.append(f"    Поле в dogovor: {mapped} (рекомендуется)")
        lines.append("")

    lines.append("Требования к разработке:")
    lines.append("-" * 40)
    lines.append("1. Создать или доработать функцию импорта данных из Excel/CSV в программу dogovor.")
    lines.append("2. Сопоставить колонки файла с полями модели Contract согласно таблице выше.")
    lines.append("3. Если поле не найдено в модели Contract — создать новое или предложить адаптацию.")
    lines.append("4. Обеспечить обработку пропусков (пустых ячеек), дублей, некорректных форматов.")
    lines.append("5. После импорта вывести отчёт: сколько записей добавлено, сколько пропущено, с какими ошибками.")
    lines.append("")
    lines.append(f"Всего колонок в файле: {len(columns)}, отобрано для импорта: {len(enabled)}.")

    task_text = '\n'.join(lines)

    return jsonify({
        'task_text': task_text,
        'enabled_columns': len(enabled),
        'total_columns': len(columns),
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
