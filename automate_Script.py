import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import time
import requests
import re
from bs4 import BeautifulSoup

# --- 1. CONFIGURE YOUR DETAILS ---
YOUR_EMAIL = "your_email@gmail.com"  # Your Gmail address
YOUR_PASSWORD = "your_16_character_app_password"  # The App Password you generated
YOUR_NAME = "Brijesh Kumar"  # Your full name

# --- 2. PREPARE YOUR ATTACHMENTS ---
# Make sure these files are in the same folder as the script
RESUME_FILE = "resume.pdf"
UNDERGRAD_TRANSCRIPT_FILE = "undergrad_transcript.pdf"
LOR_FILE = "letter_of_recommendation.pdf"
attachments = [RESUME_FILE, UNDERGRAD_TRANSCRIPT_FILE, LOR_FILE]

# --- 3. CUSTOMIZE YOUR EMAIL ---
SUBJECT = "Inquiry Regarding Grader Position - Fall 2025"

# You can customize the body of the email here.
# The {professor_name} part will be automatically replaced with the name from the email address.
BODY_TEMPLATE = """
Dear Professor {professor_name},

My name is {your_name}, a Master's student in Computer Science. My academic record reflects a strong commitment to excellence; during my undergraduate studies, I earned a CGPA of 9.0/10 (evaluated as 3.98/4.0 by WES) and graduated with a rank of 50 out of 1500 students. I have continued this focus at ASU, achieving a 3.8 GPA in my first semester. I am writing to express my keen interest in a Grader position within the School of Mathematical and Natural Sciences for the upcoming Fall 2025 semester.

My most relevant experience comes from serving as a Teaching Assistant for a large, proof-based Discrete Mathematics course, where I was responsible for grading all assignments and exams for over 120 students. This, combined with my current role as a volunteer Data Analyst at the Biodesign Institute—where I apply statistical methods to biological data and create visualizations in Python and R—and my 1.5+ years of software development experience, has given me a robust quantitative and analytical skill set that I believe would be valuable in supporting your teaching.

I am confident in my ability to provide high-quality, reliable grading support. My resume, undergrad transcript, and a letter of recommendation from my former supervising professor are attached for your detailed review.

Thank you for your time and consideration.

Sincerely,
{your_name}
"""


def extract_emails_from_url(url):
    """Fetches a URL and extracts all unique email addresses."""
    print(f"\nAttempting to scrape emails from: {url}")
    try:
        # Note: You might need to install these libraries:
        # pip install requests beautifulsoup4
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()

        # Regex to find email addresses
        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)

        # Return a list of unique emails
        unique_emails = sorted(list(set(emails)))
        print(f"Found {len(unique_emails)} unique email addresses.")
        return unique_emails

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during scraping: {e}")
        return None


def send_email(recipient_email, subject, body, attachment_paths):
    """Function to send an email with attachments."""
    try:
        # Create the email message
        msg = MIMEMultipart()
        msg["From"] = f"{YOUR_NAME} <{YOUR_EMAIL}>"
        msg["To"] = recipient_email
        msg["Subject"] = subject

        # Attach the body of the email
        msg.attach(MIMEText(body, "plain"))

        # Attach files
        for file_path in attachment_paths:
            try:
                with open(file_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {file_path}",
                )
                msg.attach(part)
            except FileNotFoundError:
                print(
                    f"Error: Attachment file not found at '{file_path}'. Skipping this attachment."
                )
                continue

        # Send the email
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(YOUR_EMAIL, YOUR_PASSWORD)
        text = msg.as_string()
        server.sendmail(YOUR_EMAIL, recipient_email, text)
        server.quit()
        print(f"Successfully sent email to {recipient_email}")
        return True

    except Exception as e:
        print(f"Failed to send email to {recipient_email}. Error: {e}")
        return False


# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    # --- 4. GET RECIPIENT LIST FROM URL ---
    target_url = input("Please enter the URL to extract emails from: ")

    email_list = extract_emails_from_url(target_url)

    if not email_list:
        print("No emails found or an error occurred. Exiting program.")
    else:
        print("\nStarting email automation process...")
        # Loop through each email address and send the email
        for email in email_list:
            # Personalize the email body
            # This extracts the name before the '@' and capitalizes it.
            # e.g., "max.underwood@asu.edu" becomes "Max Underwood"
            try:
                name_part = email.split("@")[0]
                professor_name = name_part.replace(".", " ").replace("_", " ").title()
            except:
                professor_name = "Hiring Committee"  # A fallback name

            body = BODY_TEMPLATE.format(
                professor_name=professor_name, your_name=YOUR_NAME
            )

            send_email(email, SUBJECT, body, attachments)

            # Wait for a few seconds between emails to avoid being flagged as spam
            print("Waiting 5 seconds before next email...")
            time.sleep(5)

        print("\nEmail automation process finished.")
