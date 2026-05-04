# Database Explanation - How It Works

## 📊 Your Database Structure

### 1. **Users Table** (from `users/models.py`)
Stores user accounts and authentication info.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Unique ID (auto-generated) |
| username | String | Login username |
| email | String | User email |
| password | String | Encrypted password |
| role | String | 'user', 'moderator', or 'admin' |
| reputation | Integer | User reputation points |
| created_at | DateTime | When account was created |

**Example Data:**
```
id | username | email           | role  | reputation | created_at
1  | john     | john@email.com  | user  | 100        | 2024-01-15
2  | admin    | admin@email.com | admin | 1000       | 2024-01-10
```

---

### 2. **AnalysisResult Table** (from `analysis/models.py`)
Stores image/video/text analysis results.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Unique ID |
| user_id | Integer | Foreign Key → Users table |
| file_type | String | 'image', 'video', or 'text' |
| file_path | String | Path to uploaded file |
| verdict | String | 'real', 'fake', or 'uncertain' |
| confidence | Float | 0.0 to 1.0 (e.g., 0.95 = 95%) |
| heatmap_path | String | Path to Grad-CAM heatmap image |
| metadata | JSON | Extra data (probabilities, file size, etc.) |
| explanation | Text | Human-readable explanation |
| created_at | DateTime | When analysis was done |

**Example Data:**
```
id | user_id | file_type | verdict | confidence | created_at
1  | 1       | image     | fake    | 0.95       | 2024-01-20 10:30
2  | 1       | image     | real    | 0.87       | 2024-01-20 11:15
3  | 2       | video     | fake    | 0.92       | 2024-01-20 12:00
```

---

### 3. **ForumThread Table** (from `forum/models.py`)
Stores forum discussion threads.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Unique ID |
| title | String | Thread title |
| content | Text | Thread content |
| author_id | Integer | Foreign Key → Users table |
| votes | Integer | Upvotes/downvotes |
| view_count | Integer | How many times viewed |
| is_pinned | Boolean | Pinned to top? |
| created_at | DateTime | When created |

---

### 4. **ForumPost Table** (from `forum/models.py`)
Stores replies to forum threads.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Unique ID |
| thread_id | Integer | Foreign Key → ForumThread table |
| author_id | Integer | Foreign Key → Users table |
| content | Text | Post content |
| votes | Integer | Upvotes/downvotes |
| created_at | DateTime | When posted |

---

### 5. **PDAEntry Table** (from `archive/models.py`)
Stores Public Deepfake Archive entries.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Unique ID |
| title | String | Entry title |
| description | Text | Description |
| category | String | 'political', 'celebrity', etc. |
| file_path | String | Path to file |
| analysis_data | JSON | Analysis results |
| submitted_by_id | Integer | Foreign Key → Users table |
| approved | Boolean | Moderator approved? |
| created_at | DateTime | When submitted |

---

## 🔗 How Tables Connect (Relationships)

### Foreign Keys:
- **AnalysisResult.user_id** → **Users.id** (Many-to-One)
  - One user can have many analysis results
  - Each analysis result belongs to one user

- **ForumThread.author_id** → **Users.id** (Many-to-One)
  - One user can create many threads
  - Each thread has one author

- **ForumPost.thread_id** → **ForumThread.id** (Many-to-One)
  - One thread can have many posts
  - Each post belongs to one thread

---

## 💾 How Data Flows

### Example: User Uploads Image

```
1. User uploads image via API
   ↓
2. views.py receives request
   ↓
3. ML model analyzes image
   ↓
4. views.py creates AnalysisResult:
   
   AnalysisResult.objects.create(
       user=request.user,           # Links to Users table
       file_type='image',
       verdict='fake',
       confidence=0.95,
       ...
   )
   ↓
5. Django ORM converts to SQL:
   INSERT INTO analysis_analysisresult 
   (user_id, file_type, verdict, confidence, ...)
   VALUES (1, 'image', 'fake', 0.95, ...)
   ↓
6. PostgreSQL stores data in database
   ↓
7. Response sent back to user
```

---

## 🛠️ How to Use Database

### In Python Code (Django ORM):

```python
# CREATE (Save new data)
result = AnalysisResult.objects.create(
    user=user,
    verdict='fake',
    confidence=0.95
)

# READ (Get data)
all_results = AnalysisResult.objects.all()
user_results = AnalysisResult.objects.filter(user=user)
fake_results = AnalysisResult.objects.filter(verdict='fake')

# UPDATE (Modify data)
result.confidence = 0.98
result.save()

# DELETE (Remove data)
result.delete()
```

---

## 📍 Database Location

### If using Docker:
- **Location:** Inside Docker container
- **Data stored in:** Docker volume `postgres_data`
- **Access:** Through Django ORM (you don't directly access files)

### If using local PostgreSQL:
- **Location:** On your computer
- **Default location (Windows):** `C:\Program Files\PostgreSQL\15\data\`
- **Access:** Through Django ORM or pgAdmin tool

### If using SQLite (simpler option):
- **Location:** `dmi-backend/db.sqlite3` (single file)
- **No installation needed!**
- **Perfect for development/testing**

---

## 🚀 Quick Start Options

### Option 1: Use Docker (Easiest)
```bash
docker-compose up -d
```
This automatically:
- Downloads PostgreSQL
- Creates database
- Sets up everything

### Option 2: Use SQLite (Simplest for Development)
Change `settings.py`:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```
No installation needed! Just works!

### Option 3: Install PostgreSQL Locally
1. Download from: https://www.postgresql.org/download/
2. Install
3. Create database: `createdb dmi_db`
4. Update settings.py with your credentials

---

## ✅ Summary

- **Database Type:** PostgreSQL (can switch to SQLite for simplicity)
- **Is it paid?** NO! PostgreSQL is 100% FREE
- **Where is it?** In Docker container OR on your computer
- **How does it work?** Django ORM converts Python code to SQL
- **What's stored?** Users, analysis results, forum posts, archive entries

