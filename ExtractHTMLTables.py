import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4.element import Tag
import re


def extract_html_tables(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = []

    item_regex = re.compile(r'ITEM\s+(\d+[A-Z]?)\.?\s*[:\-]?\s*(.*)', re.IGNORECASE)
    current_section = None

    def clean_text(text):
        return text.replace('\xa0', ' ').strip()

    def merge_currency_cells(row):
        merged = []
        skip = False
        for i in range(len(row)):
            if skip:
                skip = False
                continue
            if row[i] == "$" and i + 1 < len(row):
                merged.append(f"${row[i + 1]}")
                skip = True
            else:
                merged.append(row[i])
        return merged

    def merge_percent_cells(row):
        merged = []
        skip = False
        for i in range(len(row)):
            if skip:
                skip = False
                continue
            current = row[i].strip()
            next_cell = row[i + 1].strip() if i + 1 < len(row) else ""
            if re.match(r"^-?\(?\d+(\.\d+)?\)?$", current) and next_cell == "%":
                merged.append(f"{current}%")
                skip = True
            else:
                merged.append(current)
        return merged

    def get_closest_span(element, direction="prev"):
        skip_phrases = [
            "copyright", "all rights reserved", "sec filings", "terms of service",
            "privacy policy", "form 10-k", "form 10-q", "form 8-q", "©", "&#169;"
        ] #Can add more phrases to the list to broaden the span skip

        next_span = element.find_previous("span") if direction == "prev" else element.find_next("span")

        while next_span:
            text = clean_text(next_span.get_text())
            if text:
                lower_text = text.lower()
                if any(phrase in lower_text for phrase in skip_phrases):
                    next_span = next_span.find_previous("span") if direction == "prev" else next_span.find_next("span")
                    continue
                return text

            next_span = next_span.find_previous("span") if direction == "prev" else next_span.find_next("span")

        return ""

    def is_width_only_row(tr):
        tds = tr.find_all("td")
        return all(td.get("style", "").startswith("width") and not td.get_text(strip=True) for td in tds)

    def get_table_header_and_data(tr_elements):
        header_row = None
        data_rows = []

        for tr in tr_elements:
            if is_width_only_row(tr):
                continue

            cells = tr.find_all(["td", "th"])
            texts = [clean_text(td.get_text()) for td in cells]

            # Header row = row with bold spans
            if not header_row and any("font-weight:700" in span.get("style", "") for td in cells for span in td.find_all("span")):
                header_row = texts
                continue

            row = merge_currency_cells(texts)
            row = merge_percent_cells(row)
            if any(cell.strip() for cell in row):
                data_rows.append(row)

        return header_row, data_rows

    last_table = None

    for tag in soup.find_all(["span", "table"]):
        # 🔹 Detect section header from span
        if tag.name == "span":
            text = tag.get_text(strip=True)
            style = tag.get("style", "")
            if item_regex.search(text) and "font-weight:700" in style:
                match = item_regex.search(text)
                if match:
                    number = match.group(1)
                    title = match.group(2)
                    current_section = f"Item {number}: {title}" if title else f"Item {number}"

        # 🔹 Table extraction starts here
        if tag.name == "table":
            trs = tag.find_all("tr")
            header, data_rows = get_table_header_and_data(trs)

            if not data_rows:
                continue

            # Use default name for empty headers
            if header and not header[0].strip():
                header[0] = "Category"

            # Remove empty headers and fill remaining
            header = [cell.strip() for cell in header if cell.strip()] if header else []
            max_cols = max(len(header), max((len(row) for row in data_rows), default=0))

            while len(header) < max_cols:
                header.append(f"col_{len(header)}")

            padded_data = []
            for row in data_rows:
                shifted_row = [cell.strip() for cell in row if cell.strip()]
                while len(shifted_row) < max_cols:
                    shifted_row.append("")
                padded_data.append(shifted_row)

            df = pd.DataFrame(padded_data, columns=header if header else [f"col_{i}" for i in range(max_cols)])

            if df.dropna(how="all").shape[0] <= 1:
                continue

            prev_span = get_closest_span(tag, "prev")
            next_span = get_closest_span(tag, "next")

            # 🔁 Merge logic starts here
            should_merge = False
            if last_table:
                same_section = last_table["section_header"] == (current_section or "Unknown")
                
                # ✅ Simplified logic: if all headers are "col_*", it's a follow-up table
                headers_are_all_generic = all(h.startswith("col_") for h in header)
                headers_missing = headers_are_all_generic or not header

                if same_section and headers_missing:
                    should_merge = True

            if should_merge:
                print(f"[MERGE] Merged table into previous — section: {current_section}")


                # 🔧 Adjust columns if necessary (padding or trimming)
                prev_cols = list(last_table["table"].columns)
                col_diff = len(prev_cols) - len(df.columns)

                if col_diff > 0:
                    for i in range(col_diff):
                        df[f"pad_col_{i}"] = ""
                elif col_diff < 0:
                    df = df.iloc[:, :len(prev_cols)]

                df.columns = prev_cols
                last_table["table"] = pd.concat([last_table["table"], df], ignore_index=True)
                last_table["num_rows"] = len(last_table["table"])
                continue



            # ➕ Add as new table
            current_table = {
                "table": df,
                "prev_span": prev_span,
                "next_span": next_span,
                "section_header": current_section or "Unknown",
                "num_rows": len(df),
                "num_cols": len(df.columns)
            }

            tables.append(current_table)
            last_table = current_table

    return tables