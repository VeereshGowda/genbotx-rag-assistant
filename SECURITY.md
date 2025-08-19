# GenBotX Security Configuration Guide

## Overview

This guide provides comprehensive security configuration recommendations for deploying GenBotX in production environments. Security considerations are essential for protecting sensitive documents, maintaining user privacy, and ensuring system integrity.

## Environment Variables Security

### Sensitive Configuration Management

GenBotX supports secure configuration through environment variables for sensitive settings:

```bash
# Copy example configuration
cp .env.example .env

# Edit with your secure values
nano .env
```

**Critical Security Variables:**
```bash
# Generate a strong secret key for session management
SECRET_KEY=your_very_strong_secret_key_here_minimum_32_characters

# Restrict allowed hosts for production deployment
ALLOWED_HOSTS=yourdomain.com,10.0.0.1,localhost

# Configure secure logging level
LOG_LEVEL=WARNING  # Use WARNING or ERROR in production

# Secure upload directory permissions
UPLOAD_DIRECTORY=./secure_uploads
```

### Secret Key Generation

**Generate Secure Secret Keys:**
```python
# Use this Python script to generate secure keys
import secrets
print(secrets.token_urlsafe(32))
```

## File Upload Security

### Upload Restrictions

**File Type Validation:**
GenBotX implements strict file type validation to prevent security vulnerabilities:

- Only PDF, DOCX, and TXT files are accepted
- File content validation beyond extension checking
- Maximum file size limits (configurable via MAX_UPLOAD_SIZE_MB)

**Directory Security:**
```bash
# Set secure permissions for upload directories
chmod 755 uploads/
chmod 755 documents/

# Ensure restricted access to sensitive directories
chmod 700 vector_store/
chmod 700 logs/
```

### Content Sanitization

The system implements automatic content sanitization:
- Malicious script detection and removal
- Binary content validation
- Encoding verification and normalization

## Access Control

### Network Security

**Firewall Configuration:**
```bash
# Allow only necessary ports
# Streamlit default: 8501
# Ollama default: 11434 (localhost only recommended)

# Example UFW rules for Ubuntu
sudo ufw allow 8501/tcp
sudo ufw enable
```

**Reverse Proxy Setup (Recommended):**
Use nginx or Apache as a reverse proxy for production deployment:

```nginx
# nginx configuration example
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Authentication Integration

**Basic Authentication Setup:**
For simple authentication, integrate with nginx basic auth:

```nginx
# Add to nginx configuration
auth_basic "GenBotX Access";
auth_basic_user_file /etc/nginx/.htpasswd;
```

**Enterprise Authentication:**
For enterprise deployments, consider integration with:
- LDAP/Active Directory
- OAuth 2.0 providers
- SAML SSO solutions

## Data Protection

### Document Security

**Encryption at Rest:**
```bash
# Use encrypted storage for sensitive documents
# Example with dm-crypt on Linux
sudo cryptsetup luksFormat /dev/sdx
sudo cryptsetup luksOpen /dev/sdx encrypted_storage
```

**Access Logging:**
Enable comprehensive access logging:

```yaml
# In config.yaml
logging:
  level: INFO
  access_logging: true
  log_queries: true  # Set to false for sensitive environments
  log_uploads: true
```

### Vector Store Security

**ChromaDB Security:**
- Store vector database in secure directory with restricted permissions
- Regular backup with encryption
- Monitor for unauthorized access patterns

## Monitoring and Auditing

### Security Monitoring

**Log Analysis:**
Monitor logs for security events:
- Unusual upload patterns
- Repeated failed queries
- Large file uploads
- Resource exhaustion attempts

**System Monitoring:**
```bash
# Monitor system resources
htop
iostat
df -h

# Check for suspicious processes
ps aux | grep -E "(python|streamlit|ollama)"
```

### Audit Trail

**Query Auditing:**
Configure query logging for compliance:

```yaml
# Enhanced logging configuration
logging:
  audit_queries: true
  audit_uploads: true
  audit_access: true
  retention_days: 90
```

## Deployment Security

### Container Security

**Docker Security Best Practices:**
```dockerfile
# Use non-root user
RUN adduser --disabled-password genbotx
USER genbotx

# Minimal base image
FROM python:3.12-slim

# Security scanning
RUN apt-get update && apt-get upgrade -y
```

### Cloud Deployment

**AWS Security Considerations:**
- Use IAM roles with minimal permissions
- Enable CloudTrail for audit logging
- Configure VPC with private subnets
- Use Application Load Balancer with SSL termination

**Azure Security Considerations:**
- Use Managed Identity for authentication
- Enable Azure Monitor for logging
- Configure Network Security Groups
- Use Application Gateway with WAF

## Incident Response

### Security Incident Procedures

**Immediate Response:**
1. Isolate affected systems
2. Preserve evidence and logs
3. Assess scope of compromise
4. Notify relevant stakeholders

**Recovery Procedures:**
1. Clean affected systems
2. Update security configurations
3. Review and update access controls
4. Conduct post-incident review

### Backup and Recovery

**Secure Backup Strategy:**
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf "genbotx_backup_$DATE.tar.gz" \
    vector_store/ \
    config.yaml \
    documents/ \
    --exclude='logs/*'

# Encrypt backup
gpg --symmetric --cipher-algo AES256 "genbotx_backup_$DATE.tar.gz"
```

## Compliance Considerations

### Data Privacy

**GDPR Compliance:**
- Implement data subject rights (access, deletion)
- Maintain data processing records
- Ensure lawful basis for processing
- Implement privacy by design

**HIPAA Compliance (if applicable):**
- Encrypt all PHI at rest and in transit
- Implement access controls and audit logs
- Sign Business Associate Agreements
- Regular security risk assessments

### Industry Standards

**ISO 27001 Alignment:**
- Implement information security management system
- Regular risk assessments
- Security awareness training
- Incident management procedures

## Security Checklist

### Pre-Deployment Security Review

- [ ] Strong secret keys generated and configured
- [ ] File upload restrictions implemented
- [ ] Network access controls configured
- [ ] Logging and monitoring enabled
- [ ] Backup and recovery procedures tested
- [ ] Security incident response plan documented
- [ ] Regular security updates scheduled
- [ ] User access controls implemented
- [ ] Data encryption configured where required
- [ ] Compliance requirements addressed

### Regular Security Maintenance

- [ ] Weekly log review and analysis
- [ ] Monthly security updates and patches
- [ ] Quarterly access control review
- [ ] Annual security assessment and penetration testing
- [ ] Continuous monitoring of security advisories

This security guide provides a foundation for secure GenBotX deployment. Organizations should adapt these recommendations based on their specific security requirements, compliance obligations, and risk tolerance.
