import PyPDF2
from docx import Document
import logging
from settings import MAX_TEXT_LENGTH

logger = logging.getLogger(__name__)

class FileProcessor:
    def process_file(self, uploaded_file):
        """Process uploaded file and extract text content"""
        try:
            file_content = ""
            file_type = uploaded_file.type
            file_name = uploaded_file.name
            
            if file_type == "text/plain":
                file_content = str(uploaded_file.read(), "utf-8")
                
            elif file_type == "application/pdf":
                # Process PDF
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text_parts = []
                
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(text)
                
                file_content = "\n".join(text_parts)
                
            elif file_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword"
            ]:
                # Process DOCX
                doc = Document(uploaded_file)
                text_parts = []
                
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text_parts.append(paragraph.text)
                
                file_content = "\n".join(text_parts)
                
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                return None
            
            # Clean and limit content
            if file_content.strip():
                # Clean text
                lines = file_content.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line:
                        cleaned_lines.append(line)
                
                file_content = "\n".join(cleaned_lines)
                
                # Limit content length
                if len(file_content) > MAX_TEXT_LENGTH:
                    file_content = file_content[:MAX_TEXT_LENGTH] + "...[truncated]"
                
                logger.info(f"Successfully processed {file_name}: {len(file_content)} characters")
                
                return {
                    'name': file_name,
                    'content': file_content,
                    'type': file_type
                }
            else:
                logger.warning(f"No content extracted from {file_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
            return None

file_processor = FileProcessor()