import os
from typing import List
import PyPDF2
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Process different types of documents and extract text"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def process_document(self, file_path: str, filename: str) -> List[str]:
        """Process a document based on its file extension"""
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self._process_pdf(file_path)
        elif file_extension == 'txt':
            return self._process_txt(file_path)
        elif file_extension == 'csv':
            return self._process_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def process_text(self, text: str, source_name: str) -> List[str]:
        """Process raw text and split into chunks"""
        if not text.strip():
            return []
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def _process_pdf(self, file_path: str) -> List[str]:
        """Extract text from PDF file with multiple fallback methods"""
        text = ""
        
        # Method 1: Try PyPDF2 with character cleaning
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Clean the text to remove problematic characters
                            page_text = self._clean_pdf_text(page_text)
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                        continue
                        
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
        
        # Method 2: Try pdfplumber if PyPDF2 failed or produced no text
        if not text.strip():
            try:
                import pdfplumber
                print("Trying pdfplumber for PDF extraction...")
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += self._clean_pdf_text(page_text) + "\n"
                        except Exception as e:
                            print(f"pdfplumber: Could not extract from page {page_num + 1}: {e}")
                            continue
            except ImportError:
                print("pdfplumber not available")
            except Exception as e:
                print(f"pdfplumber extraction failed: {e}")
        
        # Method 3: Binary extraction fallback
        if not text.strip():
            print("Trying binary extraction fallback...")
            text = self._extract_pdf_binary_fallback(file_path)
        
        # Method 4: OCR-like text pattern extraction
        if not text.strip():
            print("Trying pattern-based text extraction...")
            text = self._extract_pdf_pattern_fallback(file_path)
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF. The PDF might be image-based, corrupted, or password-protected. Try converting it to a text-based PDF first.")
        
        return self.text_splitter.split_text(text)
    
    def _clean_pdf_text(self, text: str) -> str:
        """Clean PDF text to handle encoding issues"""
        if not text:
            return ""
        
        # Remove problematic Unicode characters
        problematic_chars = [
            '\udbef', '\udcef',  # Surrogate characters
            '\ufeff',            # BOM
            '\u200b', '\u200c', '\u200d',  # Zero-width characters
            '\u2028', '\u2029',  # Line/paragraph separators
        ]
        
        for char in problematic_chars:
            text = text.replace(char, '')
        
        # Handle encoding issues
        try:
            # Try to encode/decode to clean up encoding issues
            text = text.encode('utf-8', 'ignore').decode('utf-8')
        except:
            # Fallback: keep only printable ASCII characters
            text = ''.join(char for char in text if ord(char) < 128 and (char.isprintable() or char.isspace()))
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_pdf_binary_fallback(self, file_path: str) -> str:
        """Binary extraction fallback for problematic PDFs"""
        try:
            with open(file_path, 'rb') as file:
                content = file.read()
            
            # Try to decode as latin-1 first (preserves byte values)
            try:
                content_str = content.decode('latin-1', errors='ignore')
            except:
                content_str = content.decode('utf-8', errors='ignore')
            
            # Look for text between common PDF text markers
            import re
            
            # Find text streams in PDF
            text_patterns = []
            
            # Pattern 1: Text between BT and ET markers
            bt_et_pattern = r'BT\s+(.*?)\s+ET'
            matches = re.findall(bt_et_pattern, content_str, re.DOTALL)
            for match in matches:
                # Extract text from PDF text commands
                text_commands = re.findall(r'\((.*?)\)', match)
                text_patterns.extend(text_commands)
            
            # Pattern 2: Look for readable text sequences
            readable_text = re.findall(r'[A-Za-z0-9\s\.,;:!?\-]{20,}', content_str)
            text_patterns.extend(readable_text)
            
            # Combine and clean
            extracted_text = ' '.join(text_patterns)
            return self._clean_pdf_text(extracted_text) if extracted_text else ""
            
        except Exception as e:
            print(f"Binary extraction failed: {e}")
            return ""
    
    def _extract_pdf_pattern_fallback(self, file_path: str) -> str:
        """Pattern-based extraction for difficult PDFs"""
        try:
            with open(file_path, 'rb') as file:
                content = file.read()
            
            # Convert to string with error handling
            content_str = content.decode('latin-1', errors='ignore')
            
            import re
            
            # Look for various text patterns
            patterns = [
                r'/Title\s*\((.*?)\)',  # PDF title
                r'/Subject\s*\((.*?)\)',  # PDF subject
                r'/Author\s*\((.*?)\)',  # PDF author
                r'>\s*([A-Za-z][A-Za-z0-9\s\.,;:!?\-]{10,})\s*<',  # Text between angle brackets
                r'\]\s*([A-Za-z][A-Za-z0-9\s\.,;:!?\-]{10,})\s*\[',  # Text between square brackets
            ]
            
            extracted_texts = []
            for pattern in patterns:
                matches = re.findall(pattern, content_str, re.IGNORECASE)
                extracted_texts.extend(matches)
            
            # Also try to find any readable text sequences
            readable_sequences = re.findall(r'[A-Za-z][A-Za-z0-9\s\.,;:!?\-]{15,}', content_str)
            extracted_texts.extend(readable_sequences[:10])  # Limit to avoid noise
            
            combined_text = ' '.join(extracted_texts)
            return self._clean_pdf_text(combined_text) if combined_text else ""
            
        except Exception as e:
            print(f"Pattern extraction failed: {e}")
            return ""
    
    def _process_txt(self, file_path: str) -> List[str]:
        """Process plain text file with multiple encoding support"""
        try:
            # Try multiple encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'ascii']
            text = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                # Last resort: read as binary and decode with error handling
                with open(file_path, 'rb') as file:
                    raw_content = file.read()
                    text = raw_content.decode('utf-8', errors='ignore')
            
            if not text.strip():
                raise ValueError("The text file is empty")
            
            return self.text_splitter.split_text(text)
            
        except Exception as e:
            raise ValueError(f"Error processing text file: {str(e)}")
    
    def _process_csv(self, file_path: str) -> List[str]:
        """Process CSV file with encoding detection"""
        try:
            # Try multiple encodings for CSV
            encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception as e:
                    if "encoding" not in str(e).lower():
                        raise
            
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding")
            
            if df.empty:
                raise ValueError("The CSV file is empty")
            
            # Convert DataFrame to text representation
            text_parts = []
            
            # Add column headers
            text_parts.append("CSV Data Structure:")
            text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
            text_parts.append(f"Total rows: {len(df)}")
            text_parts.append("\nData:")
            
            # Convert each row to text (limit to first 100 rows for performance)
            max_rows = min(100, len(df))
            for index, row in df.head(max_rows).iterrows():
                row_text = f"Row {index + 1}: "
                row_items = []
                for col, value in row.items():
                    # Clean the value to avoid encoding issues
                    if pd.notna(value):
                        clean_value = str(value).encode('ascii', 'ignore').decode('ascii')
                    else:
                        clean_value = 'N/A'
                    row_items.append(f"{col}: {clean_value}")
                row_text += ", ".join(row_items)
                text_parts.append(row_text)
            
            if len(df) > max_rows:
                text_parts.append(f"\n... and {len(df) - max_rows} more rows")
            
            # Join all parts
            full_text = "\n".join(text_parts)
            
            return self.text_splitter.split_text(full_text)
            
        except Exception as e:
            raise ValueError(f"Error processing CSV file: {str(e)}")
    
    def get_file_info(self, file_path: str) -> dict:
        """Get basic information about a file"""
        try:
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            file_extension = file_name.split('.')[-1].lower()
            
            return {
                "name": file_name,
                "size": file_size,
                "extension": file_extension,
                "size_mb": round(file_size / (1024 * 1024), 2)
            }
        except Exception as e:
            return {"error": str(e)}