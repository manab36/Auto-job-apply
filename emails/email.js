function sendEmailsWithAttachments() {
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    const rows = sheet.getDataRange().getValues();
    const header = rows[0];
  
    const emailCol = header.indexOf("Email");
    const subjectCol = header.indexOf("Subject");
    const messageCol = header.indexOf("Message");
    const attachmentCol = header.indexOf("Attachment");
  
    for (let i = 1; i < rows.length; i++) {
      const email = rows[i][emailCol];
      const subject = rows[i][subjectCol];
      const message = rows[i][messageCol];
      const attachmentFileId = rows[i][attachmentCol];
  
      // Log recipient details
      console.log(`Processing row ${i}: Email=${email}, Subject=${subject}, Attachment ID=${attachmentFileId}`);
  
      if (email && subject && message && attachmentFileId) {
        try {
          const file = DriveApp.getFileById(attachmentFileId);
          GmailApp.sendEmail(email, subject, message, {
            attachments: [file.getAs(MimeType.PDF)]
          });
          console.log(`Email sent to ${email}`);
        } catch (e) {
          console.error(`Error sending email to ${email}: ${e.message}`);
        }
      } else {
        console.warn(`Skipping row ${i}: Missing required data.`);
      }
    }
  }
  