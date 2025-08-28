// server.js - Node.js Authentication Backend

const express = require('express');
require('dotenv').config(); 
const mysql = require('mysql2/promise'); 
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');

// --- Configuration ---
const app = express();
const PORT = process.env.PORT || 5000;
const JWT_SECRET = process.env.JWT_SECRET || 'your_jwt_secret_key';
const DB_CONFIG = {
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || 'password',
    database: process.env.DB_DATABASE || 'nyay_ai_db',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
};

// --- Middleware ---
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// ✅ Serve static frontend files
app.use(express.static(path.join(__dirname)));

// --- Database Connection Pool ---
let pool;

async function initializeDatabase() {
    try {
        console.log("⚡ Connecting to MySQL with config:", DB_CONFIG);
        pool = mysql.createPool(DB_CONFIG);
        console.log('✅ MySQL connection pool created successfully.');
        await pool.query('SELECT 1');
        console.log('✅ Database connection tested successfully.');
    } catch (err) {
        console.error('❌ Failed to connect to MySQL database:', err.message);
        process.exit(1);
    }
}

// ✅ JWT Middleware
const authenticateToken = (req, res, next) => {
    console.log("🔑 Authenticating token...");
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1]; // "Bearer <token>"

    if (!token) {
        console.warn("⚠️ No token provided");
        return res.status(401).json({ message: 'Authentication token required.' });
    }

    jwt.verify(token, JWT_SECRET, (err, user) => {
        if (err) {
            console.error('❌ JWT verification failed:', err.message);
            return res.status(403).json({ message: 'Invalid or expired token.' });
        }
        console.log("✅ JWT verified for user:", user.user.email);
        req.user = user;
        next();
    });
};

// --- Routes ---

// API status check
app.get('/api', (req, res) => {
    console.log("📡 /api status check called");
    res.json({ message: 'Nyay AI Backend API is running!' });
});

/**
 * @route POST /api/signup
 */
app.post('/api/signup', async (req, res) => {
    const { username, email, password } = req.body;
    console.log("📝 Signup attempt:", { username, email });

    if (!username || !email || !password) {
        console.warn("⚠️ Missing signup fields");
        return res.status(400).json({ message: 'All fields (username, email, password) are required.' });
    }
    if (password.length < 8) {
        console.warn("⚠️ Password too short for email:", email);
        return res.status(400).json({ message: 'Password must be at least 8 characters long.' });
    }

    try {
        const [existingUsers] = await pool.query('SELECT id FROM users WHERE email = ?', [email]);
        if (existingUsers.length > 0) {
            console.warn("⚠️ Duplicate signup attempt for:", email);
            return res.status(409).json({ message: 'User with this email already exists.' });
        }

        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);
        console.log("🔐 Password hashed for:", email);

        const [result] = await pool.query(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            [username, email, hashedPassword]
        );

        console.log("✅ User registered with ID:", result.insertId);
        res.status(201).json({ message: 'User registered successfully!', userId: result.insertId });
    } catch (err) {
        console.error('❌ Error during signup:', err.message);
        res.status(500).json({ message: 'Server error during signup. Please try again later.' });
    }
});

/**
 * @route POST /api/login
 */
app.post('/api/login', async (req, res) => {
    const { email, password } = req.body;
    console.log("🔑 Login attempt for:", email);

    if (!email || !password) {
        console.warn("⚠️ Login failed, missing fields");
        return res.status(400).json({ message: 'Email and password are required.' });
    }

    try {
        const [users] = await pool.query('SELECT id, username, email, password FROM users WHERE email = ?', [email]);
        if (users.length === 0) {
            console.warn("⚠️ No user found for:", email);
            return res.status(401).json({ message: 'Invalid credentials.' });
        }

        const user = users[0];
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            console.warn("⚠️ Wrong password for:", email);
            return res.status(401).json({ message: 'Invalid credentials.' });
        }

        const payload = { user: { id: user.id, username: user.username, email: user.email } };
        console.log("✅ Login successful for:", email);

        jwt.sign(payload, JWT_SECRET, { expiresIn: '1h' }, (err, token) => {
            if (err) throw err;
            console.log("🔑 JWT issued for:", email);
            res.json({ message: 'Login successful!', token });
        });
    } catch (err) {
        console.error('❌ Error during login:', err.message);
        res.status(500).json({ message: 'Server error during login. Please try again later.' });
    }
});

/**
 * @route GET /api/protected
 */
app.get('/api/protected', authenticateToken, (req, res) => {
    console.log("🔒 Protected route accessed by:", req.user.user.email);
    res.json({
        message: `Welcome, ${req.user.user.username}! You have access to protected data.`,
        userData: req.user.user
    });
});

// ✅ Route for root to serve index.html
app.get('/', (req, res) => {
    console.log("📄 Serving index.html");
    res.sendFile(path.join(__dirname, 'index.html'));
});

// --- Server Start ---
async function startServer() {
    await initializeDatabase();
    app.listen(PORT, () => {
        console.log(`🚀 Server running on port ${PORT}`);
        console.log(`🌐 Frontend: http://localhost:${PORT}/`);
        console.log(`📡 API: http://localhost:${PORT}/api`);
    });
}

startServer();
