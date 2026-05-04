# 🚀 Deployment & Database Management Guide

## 📋 Table of Contents
1. [Running Locally vs Docker](#running-locally-vs-docker)
2. [What is Deployment?](#what-is-deployment)
3. [Database Management](#database-management)
4. [Authentication & Image Upload](#authentication--image-upload)
5. [Fixing "John Doe" Issue](#fixing-john-doe-issue)

---

## 🖥️ Running Locally vs Docker

### **Option 1: Running Locally (Current Setup) ✅**

**How it works:**
```powershell
# Terminal 1: Start PostgreSQL database
docker-compose up db

# Terminal 2: Start Django backend
cd dmi-backend
.\venv\Scripts\Activate.ps1
python manage.py runserver

# Terminal 3: Start Next.js frontend
cd dmi-frontend
npm run dev
```

**Pros:**
- ✅ **Easy to debug** - See errors directly in terminal
- ✅ **Fast development** - Hot reload for both frontend and backend
- ✅ **Simple setup** - No Docker knowledge needed
- ✅ **Direct file access** - Edit files directly, changes reflect immediately
- ✅ **Better for development** - Perfect for coding and testing

**Cons:**
- ❌ **Manual setup** - Need to start each service separately
- ❌ **Environment differences** - Your local machine might differ from production
- ❌ **Harder to share** - Others need same setup on their machines

---

### **Option 2: Running with Docker 🐳**

**How it works:**
```powershell
# One command starts everything
docker-compose up
```

**Pros:**
- ✅ **One command** - Everything starts together
- ✅ **Consistent environment** - Same setup everywhere
- ✅ **Easy to share** - Others can run with same Docker setup
- ✅ **Production-like** - Closer to how it runs in production
- ✅ **Isolated** - Doesn't affect your local Python/Node installations

**Cons:**
- ❌ **Slower startup** - Takes time to build containers
- ❌ **Harder to debug** - Need to check Docker logs
- ❌ **More complex** - Need to understand Docker
- ❌ **File changes** - Sometimes need to rebuild containers

---

### **Which Should You Use?**

| Scenario | Recommendation |
|----------|---------------|
| **Development (coding)** | ✅ **Local** - Faster, easier to debug |
| **Testing** | ✅ **Local** - Quick iterations |
| **Sharing with team** | ✅ **Docker** - Consistent setup |
| **Production deployment** | ✅ **Docker** - Industry standard |
| **Learning/Understanding** | ✅ **Local** - See how things work |

**For your FYP project:**
- **Development:** Use **Local** (what you're doing now) ✅
- **Final submission:** Can use **Docker** for consistency

---

## 🌐 What is Deployment?

**Deployment** = Making your website accessible to the world (not just on your computer)

### **Current State (Development):**
```
Your Computer → http://localhost:3000 (Frontend)
              → http://127.0.0.1:8000 (Backend)
```
- ❌ Only you can access it
- ❌ Stops when you close terminal
- ❌ Uses your computer's resources

### **Deployed State (Production):**
```
Internet → https://yourwebsite.com (Frontend)
        → https://api.yourwebsite.com (Backend)
```
- ✅ Anyone can access it
- ✅ Runs 24/7 on a server
- ✅ Uses cloud server resources

### **Deployment Options:**

1. **Free Options:**
   - **Vercel** (Frontend) + **Railway** (Backend) - Free tier available
   - **Netlify** (Frontend) + **Render** (Backend) - Free tier available
   - **GitHub Pages** (Frontend only)

2. **Paid Options:**
   - **AWS** (Amazon Web Services) - Pay per use
   - **Google Cloud** - Pay per use
   - **Azure** (Microsoft) - Pay per use
   - **DigitalOcean** - $5-10/month

3. **For FYP:**
   - **Vercel** (Frontend) - Free, easy, perfect for Next.js
   - **Railway** (Backend) - Free tier, easy PostgreSQL setup
   - **Total Cost: $0** ✅

### **Deployment Process:**
1. Push code to GitHub
2. Connect to deployment service
3. Configure environment variables
4. Deploy!
5. Your website is live! 🎉

---

## 💾 Database Management

### **Where is Data Stored?**

Your data is stored in a **PostgreSQL database** running in Docker:

```
Location: Docker Container (postgres:15)
Database Name: dmi_db
Username: user
Password: password
Port: 5432
```

**Physical Location:**
- Windows: `C:\Users\YourName\AppData\Local\Docker\wsl\data\ext4.vhdx`
- Linux/Mac: Docker volume on your disk

### **What Data is Stored?**

1. **Users Table** (`users_user`)
   - Username, email, password (hashed), role, reputation
   - Example: `dani9099`, `Daniqa9099#` (hashed)

2. **Analysis Results Table** (`analysis_analysisresult`)
   - Image analysis results, verdicts, confidence scores
   - Heatmap paths, file paths, metadata
   - Linked to user who uploaded

3. **Forum Tables** (`forum_thread`, `forum_post`)
   - Forum threads and posts
   - User discussions

4. **Archive Table** (`archive_pdaentry`)
   - Public deepfake archive entries

### **How to Manage Database (Admin Access)**

#### **Method 1: Django Admin Panel (Easiest) ✅**

1. **Create superuser** (if not done):
   ```powershell
   cd dmi-backend
   .\venv\Scripts\Activate.ps1
   python manage.py createsuperuser
   ```
   - Username: `admin`
   - Email: `admin@example.com`
   - Password: (choose a strong password)

2. **Start Django server:**
   ```powershell
   python manage.py runserver
   ```

3. **Open admin panel:**
   - Go to: `http://127.0.0.1:8000/admin`
   - Login with superuser credentials

4. **What you can do:**
   - ✅ View all users
   - ✅ Edit user details
   - ✅ View all analysis results
   - ✅ Delete analysis results
   - ✅ View forum posts
   - ✅ Manage archive entries
   - ✅ Change user roles (user → admin)

#### **Method 2: Direct Database Access (Advanced)**

1. **Connect to PostgreSQL:**
   ```powershell
   # Make sure Docker is running
   docker-compose up db

   # Connect to database
   docker exec -it dmi-backend-db-1 psql -U user -d dmi_db
   ```

2. **Useful SQL Commands:**
   ```sql
   -- View all users
   SELECT id, username, email, role FROM users_user;

   -- View all analysis results
   SELECT id, verdict, confidence, created_at FROM analysis_analysisresult;

   -- Count analyses per user
   SELECT user_id, COUNT(*) FROM analysis_analysisresult GROUP BY user_id;

   -- Delete a specific analysis
   DELETE FROM analysis_analysisresult WHERE id = 1;

   -- Exit
   \q
   ```

#### **Method 3: Database GUI Tools**

**pgAdmin** (Free, Recommended):
1. Download: https://www.pgadmin.org/download/
2. Connect:
   - Host: `localhost`
   - Port: `5432`
   - Database: `dmi_db`
   - Username: `user`
   - Password: `password`

**DBeaver** (Free):
1. Download: https://dbeaver.io/download/
2. Create PostgreSQL connection with same credentials

---

## 🔐 Authentication & Image Upload

### **Why Image Upload Requires Login?**

Looking at `dmi-backend/analysis/views.py`:

```python
@api_view(['POST'])
@permission_classes([IsAuthenticated])  # ← This requires login!
def analyze_image(request):
    # ... image analysis code ...
```

**The `@permission_classes([IsAuthenticated])` decorator means:**
- ✅ **Must be logged in** to upload images
- ✅ **JWT token required** in request headers
- ✅ **User must exist** in database

### **Why This Design?**

1. **Security** - Prevents abuse (spam uploads)
2. **Tracking** - Know who uploaded what
3. **User History** - Users can see their past analyses
4. **Rate Limiting** - Can limit uploads per user
5. **Data Privacy** - Each user only sees their own results

### **How It Works:**

1. **User logs in:**
   ```
   POST /api/auth/login/
   → Returns: { "access": "jwt_token_here" }
   ```

2. **Frontend stores token:**
   ```javascript
   localStorage.setItem('access_token', token);
   ```

3. **Upload image:**
   ```
   POST /api/analysis/image/
   Headers: { "Authorization": "Bearer jwt_token_here" }
   Body: { image: <file> }
   ```

4. **Backend verifies token:**
   - ✅ Valid token → Process image
   - ❌ No token → Return 401 Unauthorized

### **To Allow Anonymous Uploads (Not Recommended):**

If you want to allow uploads without login, change:

```python
# In dmi-backend/analysis/views.py
from rest_framework.permissions import AllowAny

@api_view(['POST'])
@permission_classes([AllowAny])  # ← Change this
def analyze_image(request):
    # ... rest of code ...
```

**But then:**
- ❌ Can't track who uploaded what
- ❌ No user history
- ❌ Risk of abuse

---

## 👤 Fixing "John Doe" Issue

### **Problem:**
Dashboard shows hardcoded "John Doe" instead of actual username.

### **Solution:**
1. Add `getUserProfile()` function to API client
2. Fetch user data in dashboard
3. Display actual username

**Files to modify:**
- `dmi-frontend/lib/api.ts` - Add profile function
- `dmi-frontend/app/dashboard/page.tsx` - Fetch and display user data

---

## 📊 Summary

| Question | Answer |
|----------|--------|
| **Run locally or Docker?** | ✅ **Local for development**, Docker for production |
| **What is deployment?** | Making website accessible on internet (not just localhost) |
| **Where is data stored?** | PostgreSQL database in Docker container |
| **How to manage database?** | Django Admin (`/admin`) or pgAdmin GUI |
| **Why login required?** | Security, tracking, user history |
| **Why "John Doe"?** | Hardcoded - needs to fetch real user data |

---

## 🎯 Next Steps

1. ✅ Fix "John Doe" issue (see fixes below)
2. ✅ Test image upload with login
3. ✅ Access Django admin to view database
4. ✅ Consider deployment for final submission

---

**Need help?** Check:
- Django Admin: `http://127.0.0.1:8000/admin`
- API Docs: `http://127.0.0.1:8000/api/`
- Database: Docker container `dmi-backend-db-1`

