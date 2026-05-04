# 📊 Database Management Guide - View & Manage All Data

## 🎯 Quick Answer

**Where to see and manage your database:**
- **Django Admin Panel** → `http://127.0.0.1:8000/admin` (Local)
- **Django Admin Panel** → `https://yourwebsite.com/admin` (When deployed)

**This is where you can:**
- ✅ See all new user signups
- ✅ View all analysis results
- ✅ Manage user accounts
- ✅ Delete/edit records
- ✅ See forum posts
- ✅ Manage archive entries

---

## 🚀 Step-by-Step Setup

### **Step 1: Create Admin Account (Superuser)**

You need to create an admin account first (one-time setup):

```powershell
cd dmi-backend
.\venv\Scripts\Activate.ps1
python manage.py createsuperuser
```

**What to enter:**
```
Username: admin
Email address: admin@example.com
Password: [Enter a strong password]
Password (again): [Enter again]
```

**Example:**
```
Username: admin
Email: admin@dmi.com
Password: Admin123!@#
```

✅ **Done!** You now have admin access.

---

### **Step 2: Start Django Server**

```powershell
# Make sure you're in dmi-backend directory
python manage.py runserver
```

You should see:
```
Starting development server at http://127.0.0.1:8000/
```

---

### **Step 3: Access Admin Panel**

1. **Open browser**
2. **Go to:** `http://127.0.0.1:8000/admin`
3. **Login** with your admin credentials

**You'll see the Django Admin Dashboard!** 🎉

---

## 📋 What You Can See & Manage

### **1. Users Section** 👥

**Location:** Click "Users" in admin panel

**What you see:**
- ✅ **All registered users** (including new signups!)
- ✅ Username, email, role, reputation
- ✅ Signup date (`created_at`)
- ✅ Staff status, superuser status

**What you can do:**
- ✅ **View new user signups** - See when they registered
- ✅ **Edit user details** - Change email, role, etc.
- ✅ **Delete users** - Remove accounts
- ✅ **Change user roles** - Make someone admin/moderator
- ✅ **Reset passwords** - Set new password for users
- ✅ **Search users** - Find by username or email

**Example:**
```
Username: dani9099
Email: dani@example.com
Role: user
Reputation: 0
Created: Jan 15, 2024, 10:30 AM
```

---

### **2. Analysis Results Section** 📊

**Location:** Click "Analysis Results" in admin panel

**What you see:**
- ✅ **All image analyses** from all users
- ✅ User who uploaded
- ✅ Verdict (Real/Fake/Uncertain)
- ✅ Confidence score
- ✅ Upload date
- ✅ File type

**What you can do:**
- ✅ **View all analyses** - See every image analyzed
- ✅ **Filter by verdict** - See only Fake or Real results
- ✅ **Filter by user** - See analyses from specific user
- ✅ **Filter by date** - See analyses from specific time
- ✅ **Delete analyses** - Remove unwanted results
- ✅ **Search** - Find by username

**Example:**
```
User: dani9099
File Type: image
Verdict: fake
Confidence: 0.95
Created: Jan 15, 2024, 11:45 AM
```

---

### **3. Forum Section** 💬

**Location:** Click "Forum Threads" or "Forum Posts"

**What you see:**
- ✅ All forum threads
- ✅ All forum posts
- ✅ Author, votes, creation date

**What you can do:**
- ✅ View all discussions
- ✅ Delete inappropriate posts
- ✅ Edit thread titles

---

### **4. Archive Section** 📚

**Location:** Click "PDA Entries"

**What you see:**
- ✅ Public Deepfake Archive entries
- ✅ Submitted by, approval status

**What you can do:**
- ✅ Approve/reject entries
- ✅ Edit entries

---

## 🔍 How to See New User Signups

### **Method 1: View All Users (Recommended)**

1. **Login to admin:** `http://127.0.0.1:8000/admin`
2. **Click "Users"** in the left sidebar
3. **You'll see all users** sorted by default (newest might be at top/bottom)
4. **Click "Created" column header** to sort by signup date
5. **New users appear at the top!** ✅

**To see newest first:**
- Click "Created" column header twice (descending order)
- Or use filter: "Created" → "Today" or "This week"

---

### **Method 2: Filter by Date**

1. **Go to Users section**
2. **Click "Created" filter** on the right sidebar
3. **Select:**
   - "Today" - See signups today
   - "Past 7 days" - See this week's signups
   - "This month" - See this month's signups
   - Or select specific date range

---

### **Method 3: Search for Specific User**

1. **Go to Users section**
2. **Use search box** at the top
3. **Type username or email**
4. **Press Enter**

---

## 🌐 When Website is Live (Production)

### **Accessing Admin Panel on Deployed Website**

**Same process, different URL:**

1. **Your deployed backend URL** (e.g., `https://api.yourwebsite.com`)
2. **Go to:** `https://api.yourwebsite.com/admin`
3. **Login** with same admin credentials

**Important for Production:**
- ✅ **Keep admin URL secure** - Don't share it publicly
- ✅ **Use strong password** - Very important!
- ✅ **Enable HTTPS** - Always use secure connection
- ✅ **Limit access** - Only you should have admin access

---

## 🛠️ Advanced: Custom Admin Features

### **View User's Analysis History**

1. **Go to Users section**
2. **Click on a username** (e.g., "dani9099")
3. **Scroll down** - You'll see related analysis results
4. **Click on analysis** to see full details

---

### **Bulk Actions**

1. **Select multiple users/analyses** (checkboxes)
2. **Choose action** from dropdown (e.g., "Delete selected")
3. **Click "Go"**

---

### **Export Data**

**Install django-import-export:**
```powershell
pip install django-import-export
```

Then you can export users/analyses to CSV/Excel!

---

## 📊 Database Statistics

### **Quick Stats You Can See:**

1. **Total Users:**
   - Go to Users → See count at top

2. **Total Analyses:**
   - Go to Analysis Results → See count at top

3. **Users by Role:**
   - Use filter: "Role" → Select role

4. **Analyses by Verdict:**
   - Use filter: "Verdict" → Select verdict

---

## 🔐 Security Tips

### **For Production (Live Website):**

1. **Change admin URL** (optional but recommended):
   ```python
   # In dmi_project/urls.py
   path('secret-admin/', admin.site.urls),  # Instead of 'admin/'
   ```

2. **Use environment variables** for admin credentials

3. **Enable 2FA** (Two-Factor Authentication) if possible

4. **Regular backups** - Export database regularly

5. **Monitor access** - Check who's accessing admin

---

## 🎯 Quick Reference

| Task | How to Do It |
|------|--------------|
| **See new signups** | Admin → Users → Sort by "Created" |
| **View all users** | Admin → Users |
| **View all analyses** | Admin → Analysis Results |
| **Delete user** | Admin → Users → Select → Delete |
| **Change user role** | Admin → Users → Click user → Edit role |
| **Search user** | Admin → Users → Search box |
| **Filter by date** | Admin → Users → "Created" filter |
| **View user's analyses** | Admin → Users → Click user → See related analyses |

---

## 🚨 Troubleshooting

### **Problem: Can't access admin panel**

**Solution:**
1. Make sure Django server is running
2. Check URL: `http://127.0.0.1:8000/admin` (not `/api/admin`)
3. Make sure you created superuser

---

### **Problem: "Page not found"**

**Solution:**
1. Check `dmi_project/urls.py` has: `path('admin/', admin.site.urls)`
2. Restart Django server

---

### **Problem: Can't see Users section**

**Solution:**
1. Check `users/admin.py` exists and has `@admin.register(User)`
2. Make sure `users` app is in `INSTALLED_APPS` in `settings.py`

---

### **Problem: New users not showing**

**Solution:**
1. Refresh the page
2. Check database is running: `docker-compose ps`
3. Check user actually signed up (check frontend)

---

## 📝 Example Workflow

### **Daily Admin Check:**

1. **Login to admin:** `http://127.0.0.1:8000/admin`
2. **Check Users:**
   - See new signups (sort by "Created")
   - Check for suspicious accounts
3. **Check Analysis Results:**
   - See recent analyses
   - Check for errors
4. **Check Forum:**
   - Review new posts
   - Moderate if needed

**Time: 2-5 minutes per day** ✅

---

## 🎉 Summary

**Where to manage database:**
- ✅ **Django Admin Panel** - `http://127.0.0.1:8000/admin`

**What you can do:**
- ✅ See all new user signups
- ✅ View all analysis results
- ✅ Manage users (edit, delete, change roles)
- ✅ Filter and search data
- ✅ Export data (with additional setup)

**When website is live:**
- ✅ Same process, just use your deployed URL
- ✅ `https://yourwebsite.com/admin`

**Everything is ready!** Your admin panel is already configured! 🚀

---

## 🔗 Related Files

- `dmi-backend/users/admin.py` - User admin configuration
- `dmi-backend/analysis/admin.py` - Analysis admin configuration
- `dmi-backend/dmi_project/urls.py` - Admin URL routing

---

**Need help?** Check Django admin documentation: https://docs.djangoproject.com/en/stable/ref/contrib/admin/

