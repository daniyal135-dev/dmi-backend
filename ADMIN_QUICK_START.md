# ⚡ Admin Panel Quick Start (5 Minutes)

## 🎯 Goal
Access Django Admin to see all database records, including new user signups.

---

## 📝 Step 1: Create Admin Account (One-Time)

```powershell
cd dmi-backend
.\venv\Scripts\Activate.ps1
python manage.py createsuperuser
```

**Enter:**
- Username: `admin`
- Email: `admin@example.com`
- Password: `[Your strong password]`

---

## 🚀 Step 2: Start Server

```powershell
python manage.py runserver
```

---

## 🌐 Step 3: Open Admin Panel

1. **Open browser**
2. **Go to:** `http://127.0.0.1:8000/admin`
3. **Login** with your admin credentials

---

## 👥 Step 4: See New User Signups

1. **Click "Users"** in left sidebar
2. **Click "Created"** column header to sort by date
3. **Newest users appear at top!** ✅

---

## 📊 What You Can See

| Section | What's There |
|---------|--------------|
| **Users** | All registered users, signup dates, roles |
| **Analysis Results** | All image analyses, verdicts, confidence scores |
| **Forum Threads** | All forum discussions |
| **Forum Posts** | All forum replies |
| **PDA Entries** | Archive entries |

---

## 🔍 Quick Actions

- **Search user:** Use search box at top
- **Filter by date:** Click "Created" filter → Select date range
- **View user's analyses:** Click username → See related analyses
- **Edit user:** Click username → Make changes → Save

---

## 🌐 When Website is Live

**Same process, different URL:**
- `https://yourwebsite.com/admin`
- Login with same admin credentials

---

**That's it!** You can now manage your entire database! 🎉

