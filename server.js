/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║        BharatLearn Dev Coach  –  Backend API Server  v5         ║
 * ║        Single-file Express server (server.js)                   ║
 * ╠══════════════════════════════════════════════════════════════════╣
 * ║  AI Provider: HuggingFace Inference API                         ║
 * ║  Routes:                                                         ║
 * ║    GET  /health                   liveness probe                 ║
 * ║    GET  /auth/google              start OAuth flow               ║
 * ║    GET  /auth/google/callback     OAuth callback                 ║
 * ║    GET  /auth/logout              destroy session                ║
 * ║    GET  /auth/me                  current user (or null)         ║
 * ║    POST /api/chat                 general AI chat (saves to DB)  ║
 * ║    POST /api/syllabus             syllabus → 24-week plan        ║
 * ║    POST /api/quiz                 topic → 10 quiz questions      ║
 * ║    POST /api/viva                 code file → 20 viva questions  ║
 * ║    POST /api/debug                code → hints (no solutions)    ║
 * ║    POST /api/upload               upload file to S3              ║
 * ║    GET  /api/progress/:userId     get learning progress          ║
 * ║    POST /api/progress             save learning progress         ║
 * ╚══════════════════════════════════════════════════════════════════╝
 */

import "dotenv/config";
import express      from "express";
import helmet       from "helmet";
import cors         from "cors";
import session      from "express-session";
import passport     from "passport";
import { Strategy as GoogleStrategy } from "passport-google-oauth20";
import rateLimit    from "express-rate-limit";
import morgan       from "morgan";
import Joi          from "joi";
import multer       from "multer";
import { createRequire } from "module";
import { v4 as uuid } from "uuid";

// ── CHANGED: HuggingFace replaces GoogleGenerativeAI ──────────────────────────
import Groq from "groq-sdk";

import { DynamoDBClient }                                                       from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, PutCommand, GetCommand, UpdateCommand }        from "@aws-sdk/lib-dynamodb";
import { S3Client, PutObjectCommand, GetObjectCommand }                         from "@aws-sdk/client-s3";
import { getSignedUrl }                                                         from "@aws-sdk/s3-request-presigner";
import { CloudWatchLogsClient, PutLogEventsCommand, CreateLogStreamCommand,
         DescribeLogStreamsCommand }                                             from "@aws-sdk/client-cloudwatch-logs";

const require = createRequire(import.meta.url);

// ─── Config ───────────────────────────────────────────────────────────────────
const PORT                 = process.env.PORT                || 3000;
const IS_PROD              = process.env.NODE_ENV === "production";
const FRONTEND_URL         = process.env.FRONTEND_URL        || "http://localhost:3001";
const SESSION_SECRET       = process.env.SESSION_SECRET      || "dev-only-secret-change-in-prod";
const GOOGLE_CLIENT_ID     = process.env.GOOGLE_CLIENT_ID    || "";
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || "";
const AWS_REGION           = process.env.AWS_REGION          || "us-east-1";
const S3_BUCKET            = process.env.S3_BUCKET           || "bharatlearn-storage";
const CW_LOG_GROUP         = process.env.CW_LOG_GROUP        || "/bharatlearn/api";
const CW_LOG_STREAM        = `server-${new Date().toISOString().slice(0, 10)}`;

const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.1-8b-instant";
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY || "" });

// ─── AWS Clients ──────────────────────────────────────────────────────────────
const awsCredentials = process.env.AWS_ACCESS_KEY_ID
  ? { accessKeyId: process.env.AWS_ACCESS_KEY_ID, secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY }
  : undefined;
const awsConfig = { region: AWS_REGION, ...(awsCredentials && { credentials: awsCredentials }) };

const dynamo  = DynamoDBDocumentClient.from(new DynamoDBClient(awsConfig), {
  marshallOptions: { removeUndefinedValues: true },
});
const s3      = new S3Client(awsConfig);
const cwLogs  = new CloudWatchLogsClient(awsConfig);

// ─── CloudWatch Structured Logger ────────────────────────────────────────────
let _cwSequenceToken = null;
let _cwBuffer        = [];
let _cwFlushTimer    = null;

async function _ensureLogStream() {
  try {
    await cwLogs.send(new CreateLogStreamCommand({ logGroupName: CW_LOG_GROUP, logStreamName: CW_LOG_STREAM }));
  } catch (e) {
    if (e.name !== "ResourceAlreadyExistsException") throw e;
    const res = await cwLogs.send(new DescribeLogStreamsCommand({
      logGroupName: CW_LOG_GROUP, logStreamNamePrefix: CW_LOG_STREAM,
    }));
    _cwSequenceToken = res.logStreams?.[0]?.uploadSequenceToken ?? null;
  }
}

async function _flushCWLogs() {
  if (!_cwBuffer.length) return;
  const events = _cwBuffer.splice(0);
  try {
    const res = await cwLogs.send(new PutLogEventsCommand({
      logGroupName: CW_LOG_GROUP, logStreamName: CW_LOG_STREAM,
      logEvents: events,
      ...(_cwSequenceToken && { sequenceToken: _cwSequenceToken }),
    }));
    _cwSequenceToken = res.nextSequenceToken;
  } catch (e) {
    console.error("[CW] Flush failed:", e.message);
  }
}

function log(level, event, data = {}) {
  const entry = { ts: new Date().toISOString(), level, event, ...data };
  const fn = level === "ERROR" ? console.error : level === "WARN" ? console.warn : console.log;
  fn(`[${level}] ${event}`, Object.keys(data).length ? data : "");

  if (process.env.AWS_ACCESS_KEY_ID) {
    _cwBuffer.push({ timestamp: Date.now(), message: JSON.stringify(entry) });
    if (!_cwFlushTimer) _cwFlushTimer = setInterval(_flushCWLogs, 2000);
  }
}

// ─── DynamoDB Tables ──────────────────────────────────────────────────────────
const TABLES = {
  USERS:    "bharatlearn-users",
  PROGRESS: "bharatlearn-progress",
  SESSIONS: "bharatlearn-sessions",
  PLANS:    "bharatlearn-learning-plans",
};

async function upsertUser(profile) {
  const user = {
    userId: profile.id, email: profile.emails?.[0]?.value ?? "",
    name: profile.displayName, avatar: profile.photos?.[0]?.value ?? "",
    createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
  };
  await dynamo.send(new PutCommand({ TableName: TABLES.USERS, Item: user }));
  log("INFO", "user_login", { userId: user.userId, email: user.email });
  return user;
}

async function saveChatSession(userId, sessionId, chatHistory) {
  await dynamo.send(new PutCommand({
    TableName: TABLES.SESSIONS,
    Item: { sessionId, userId, chatHistory, timestamp: new Date().toISOString() },
  }));
}

async function getProgress(userId) {
  const res = await dynamo.send(new GetCommand({ TableName: TABLES.PROGRESS, Key: { userId } }));
  return res.Item ?? null;
}

async function saveProgress(userId, topic, data) {
  await dynamo.send(new UpdateCommand({
    TableName: TABLES.PROGRESS, Key: { userId },
    UpdateExpression: "SET #topic = :data, updatedAt = :ts",
    ExpressionAttributeNames: { "#topic": topic },
    ExpressionAttributeValues: { ":data": data, ":ts": new Date().toISOString() },
  }));
}

async function saveLearningPlan(userId, plan) {
  const planId = uuid();
  await dynamo.send(new PutCommand({
    TableName: TABLES.PLANS,
    Item: { planId, userId, plan, createdAt: new Date().toISOString() },
  }));
  return planId;
}

// ─── S3 Helpers ───────────────────────────────────────────────────────────────
function getMimeType(ext) {
  const map = { pdf: "application/pdf", txt: "text/plain", py: "text/x-python",
                js: "application/javascript", ts: "text/typescript",
                java: "text/x-java-source", cpp: "text/x-c++src" };
  return map[ext] ?? "application/octet-stream";
}

async function uploadToS3(buffer, originalName, userId, folder = "uploads") {
  const ext = originalName.split(".").pop().toLowerCase();
  const key = `${folder}/${userId}/${uuid()}.${ext}`;
  await s3.send(new PutObjectCommand({
    Bucket: S3_BUCKET, Key: key, Body: buffer,
    ContentType: getMimeType(ext), Metadata: { userId, originalName },
  }));
  log("INFO", "s3_upload", { key, userId, folder, bytes: buffer.length });
  return key;
}

async function getPresignedUrl(key) {
  return getSignedUrl(s3, new GetObjectCommand({ Bucket: S3_BUCKET, Key: key }), { expiresIn: 3600 });
}

// ─── CHANGED: HuggingFace AI Helper (replaces callBedrock/Gemini) ─────────────
//
//  Uses HuggingFace's OpenAI-compatible chatCompletion endpoint.
//  Works with any instruction-tuned model on HF (Mistral, Llama, Zephyr, etc.)
//  The function signature is intentionally identical to the old callBedrock
//  so NO other code in the file needs to change.
//
async function callBedrock(system, userContent, maxTokens = 2048) {
  const messages = Array.isArray(userContent)
    ? userContent : [{ role: "user", content: userContent }];

  const t0 = Date.now();
  const response = await groq.chat.completions.create({
    model: GROQ_MODEL,
    messages: [
      { role: "system", content: system },
      ...messages.map(m => ({ role: m.role, content: m.content })),
    ],
    max_tokens: maxTokens,
    temperature: 0.7,
  });

  const text = response.choices?.[0]?.message?.content?.trim();
  if (!text) throw appError(503, "AI model returned empty response");
  log("INFO", "groq_call", { ms: Date.now() - t0, model: GROQ_MODEL });
  return text;
}

function parseAIJson(raw) {
  const clean = raw.replace(/```json[\s\S]*?```|```[\s\S]*?```/g, s => s.replace(/```json|```/gi, "").trim()).trim();
  try   { return JSON.parse(clean); }
  catch { throw appError(503, "AI returned malformed JSON – please retry"); }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function appError(status, message, code = "APP_ERROR") {
  const e = new Error(message); e.status = status; e.code = code; e.isOp = true; return e;
}
function validate(schema, data) {
  const { error, value } = schema.validate(data, { abortEarly: false, stripUnknown: true });
  if (error) throw appError(400, error.details.map(d => d.message).join("; "), "VALIDATION");
  return value;
}

// ─── Express Setup ────────────────────────────────────────────────────────────
const app = express();
app.use(helmet({ contentSecurityPolicy: IS_PROD ? undefined : false }));
app.use(cors({ origin: FRONTEND_URL, credentials: true, methods: ["GET","POST"], allowedHeaders: ["Content-Type"] }));
app.use(express.json({ limit: "32kb" }));
app.use(express.urlencoded({ extended: false, limit: "32kb" }));
app.disable("x-powered-by");
app.use(morgan(IS_PROD ? "combined" : "dev"));
app.use((req, _res, next) => { req.id = uuid(); next(); });
app.use(session({
  secret: SESSION_SECRET, resave: false, saveUninitialized: false,
  cookie: { httpOnly: true, sameSite: "lax", secure: IS_PROD, maxAge: 24 * 60 * 60 * 1000 },
}));
app.use(passport.initialize());
app.use(passport.session());

// ─── Google OAuth ─────────────────────────────────────────────────────────────
if (GOOGLE_CLIENT_ID && GOOGLE_CLIENT_SECRET) {
  passport.use(new GoogleStrategy(
    { clientID: GOOGLE_CLIENT_ID, clientSecret: GOOGLE_CLIENT_SECRET,
      callbackURL: `http://localhost:${PORT}/auth/google/callback` },
    async (_at, _rt, profile, done) => {
      try { done(null, await upsertUser(profile)); }
      catch (err) {
        log("WARN", "auth_db_fallback", { error: err.message });
        done(null, { id: profile.id, name: profile.displayName,
                     email: profile.emails?.[0]?.value, avatar: profile.photos?.[0]?.value });
      }
    }
  ));
}
passport.serializeUser((u, done)   => done(null, u));
passport.deserializeUser((u, done) => done(null, u));

// ─── Rate Limiters ────────────────────────────────────────────────────────────
const rl = (windowMs, max, msg) => rateLimit({
  windowMs, max, standardHeaders: true, legacyHeaders: false,
  message: { success: false, error: { code: "RATE_LIMIT", message: msg } },
});
app.use("/api/syllabus", rl(60_000, 6,  "AI limit: 6/min"));
app.use("/api/quiz",     rl(60_000, 8,  "AI limit: 8/min"));
app.use("/api/viva",     rl(60_000, 6,  "AI limit: 6/min"));
app.use("/api/debug",    rl(60_000, 10, "AI limit: 10/min"));
app.use("/api/chat",     rl(60_000, 15, "Chat limit: 15/min"));
app.use("/auth/",        rl(15 * 60_000, 20, "Too many auth requests. Try in 15 min."));
app.use("/api/",         rl(60_000, 60, "API limit reached"));

// ─── Multer ───────────────────────────────────────────────────────────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits:  { fileSize: 10 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const ext  = "." + file.originalname.split(".").pop().toLowerCase();
    const good = [".pdf",".txt",".py",".js",".ts",".java",".cpp",".c",".cs",".go",".rb",".php"];
    cb(good.includes(ext) ? null : appError(400, `Unsupported file type: ${ext}`), good.includes(ext));
  },
});

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get("/health", (_req, res) => {
  log("INFO", "health_check");
  res.json({ ok: true, env: process.env.NODE_ENV, ts: new Date().toISOString() });
});

app.get("/auth/google", passport.authenticate("google", { scope: ["profile","email"] }));

app.get("/auth/google/callback",
  passport.authenticate("google", { failureRedirect: `${FRONTEND_URL}/settings?error=auth_failed` }),
  (_req, res) => res.redirect(`${FRONTEND_URL}/dashboard`)
);

app.get("/auth/logout", (req, res, next) => {
  const userId = req.user?.id;
  req.logout(err => {
    if (err) return next(err);
    req.session.destroy(() => {
      log("INFO", "user_logout", { userId });
      res.clearCookie("connect.sid"); res.redirect(FRONTEND_URL);
    });
  });
});

app.get("/auth/me", (req, res) => res.json({ success: true, user: req.user ?? null }));

app.post("/api/upload", upload.single("file"), async (req, res, next) => {
  try {
    if (!req.file) throw appError(400, "No file provided");
    const userId = req.user?.id ?? "anonymous";
    const key    = await uploadToS3(req.file.buffer, req.file.originalname, userId, req.body.folder ?? "uploads");
    const url    = await getPresignedUrl(key);
    res.json({ success: true, data: { key, url, name: req.file.originalname, size: req.file.size } });
  } catch (e) { next(e); }
});

app.get("/api/progress/:userId", async (req, res, next) => {
  try { res.json({ success: true, data: await getProgress(req.params.userId) }); }
  catch (e) { next(e); }
});

const progressSchema = Joi.object({
  userId: Joi.string().required(),
  topic:  Joi.string().trim().min(1).max(200).required(),
  data:   Joi.object().required(),
});

app.post("/api/progress", async (req, res, next) => {
  try {
    const { userId, topic, data } = validate(progressSchema, req.body);
    await saveProgress(userId, topic, data);
    log("INFO", "progress_saved", { userId, topic });
    res.json({ success: true, message: "Progress saved" });
  } catch (e) { next(e); }
});

const chatSchema = Joi.object({
  message:      Joi.string().trim().min(1).max(4000).required(),
  history:      Joi.array().items(
    Joi.object({ role: Joi.string().valid("user","assistant").required(), content: Joi.string().max(4000).required() })
  ).max(20).default([]),
  systemPrompt: Joi.string().trim().max(1000).optional(),
  sessionId:    Joi.string().optional(),
});

app.post("/api/chat", async (req, res, next) => {
  try {
    const { message, history, systemPrompt, sessionId } = validate(chatSchema, req.body);
    const system   = systemPrompt || "You are Astra, a friendly AI tutor on the BharatLearn Dev Coach platform. Help CS students clearly and concisely.";
    const messages = [...history, { role: "user", content: message }];
    const reply    = await callBedrock(system, messages, 1024);
    log("INFO", "chat_message", { userId: req.user?.id ?? "anon", msgLen: message.length });
    if (req.user?.id) {
      saveChatSession(req.user.id, sessionId ?? uuid(), [...messages, { role: "assistant", content: reply }])
        .catch(e => log("WARN", "chat_save_failed", { error: e.message }));
    }
    res.json({ success: true, data: { reply } });
  } catch (e) { next(e); }
});

const syllabusBodySchema = Joi.object({ text: Joi.string().trim().min(20).max(8000).required() });

app.post("/api/syllabus", upload.single("file"), async (req, res, next) => {
  try {
    let text = "", s3Key = null;
    if (req.file) {
      const userId = req.user?.id ?? "anonymous";
      s3Key = await uploadToS3(req.file.buffer, req.file.originalname, userId, "syllabus")
        .catch(e => { log("WARN", "syllabus_s3_failed", { error: e.message }); return null; });
      text = req.file.mimetype === "application/pdf"
        ? (await require("pdf-parse")(req.file.buffer).catch(() => { throw appError(400, "PDF parse failed – try pasting text instead"); })).text?.trim() || ""
        : req.file.buffer.toString("utf-8");
    } else {
      ({ text } = validate(syllabusBodySchema, req.body));
    }
    if (text.length < 20) throw appError(400, "Syllabus content too short");
    log("INFO", "syllabus_generating", { userId: req.user?.id ?? "anon", textLen: text.length });

    const raw  = await callBedrock(
      `You are an expert curriculum designer. Convert the given syllabus into exactly a 24-week plan.
Respond ONLY with valid JSON (no markdown, no fences).
Schema: {"title":string,"description":string,"totalWeeks":24,"weeks":[{"week":number,"theme":string,"topics":string[],"tasks":string[],"milestone":string}]}
Rules: exactly 24 weeks, 2-4 topics per week, 1-2 tasks, one milestone.`,
      `Syllabus:\n${text.slice(0, 6000)}\n\nGenerate the 24-week JSON plan now.`,
      4096
    );
    const plan   = parseAIJson(raw);
    const planId = req.user?.id
      ? await saveLearningPlan(req.user.id, plan).catch(e => { log("WARN", "plan_save_failed", { error: e.message }); return null; })
      : null;
    log("INFO", "syllabus_generated", { userId: req.user?.id ?? "anon", planId });
    res.json({ success: true, data: { plan, planId, s3Key } });
  } catch (e) { next(e); }
});

const quizSchema = Joi.object({
  topic:      Joi.string().trim().min(2).max(200).required(),
  language:   Joi.string().trim().max(50).default("Python"),
  difficulty: Joi.string().valid("Beginner","Intermediate","Advanced").default("Intermediate"),
});

app.post("/api/quiz", async (req, res, next) => {
  try {
    const { topic, language, difficulty } = validate(quizSchema, req.body);
    log("INFO", "quiz_generating", { userId: req.user?.id ?? "anon", topic, difficulty });
    const raw  = await callBedrock(
      `You are an expert CS educator. Generate exactly 10 quiz questions: 4 MCQ, 3 ShortAnswer, 3 Coding.
Respond ONLY with valid JSON (no markdown).
Schema: {"topic":string,"language":string,"difficulty":string,"totalMarks":number,"questions":[{"id":number,"type":"MCQ"|"ShortAnswer"|"Coding","question":string,"difficulty":"Easy"|"Medium"|"Hard","marks":number,"options":string[]|null,"correctAnswer":string,"explanation":string,"hint":string}]}
MCQ: options A)B)C)D), correctAnswer is the letter. Marks: MCQ=1, Short=3, Coding=5.`,
      `Topic: ${topic}\nLanguage: ${language}\nDifficulty: ${difficulty}`,
      4096
    );
    const quiz = parseAIJson(raw);
    if (req.user?.id) {
      saveProgress(req.user.id, `quiz_${topic}`, { attempted: true, topic, difficulty, ts: new Date().toISOString() })
        .catch(e => log("WARN", "quiz_progress_failed", { error: e.message }));
    }
    log("INFO", "quiz_generated", { topic, count: quiz.questions?.length });
    res.json({ success: true, data: { quiz } });
  } catch (e) { next(e); }
});

app.post("/api/viva", upload.single("file"), async (req, res, next) => {
  try {
    if (!req.file) throw appError(400, "Please upload a code file");
    const code = req.file.buffer.toString("utf-8");
    if (code.trim().length < 10) throw appError(400, "File appears empty");
    const lang = req.file.originalname.split(".").pop().toLowerCase();
    if (req.user?.id) {
      uploadToS3(req.file.buffer, req.file.originalname, req.user.id, "code")
        .catch(e => log("WARN", "viva_s3_failed", { error: e.message }));
    }
    log("INFO", "viva_generating", { userId: req.user?.id ?? "anon", lang });
    const raw = await callBedrock(
      `You are an expert CS professor and viva examiner.
Analyze the code and generate exactly 20 viva questions. Never reveal answers.
Respond ONLY with valid JSON (no markdown).
Schema: {"language":string,"summary":string,"overallDifficulty":string,"evaluationCriteria":[{"criterion":string,"weight":number,"description":string}],"questions":[{"id":number,"question":string,"difficulty":"Easy"|"Medium"|"Hard","category":"Concept"|"Implementation"|"Logic"|"Debugging"|"Optimization"|"Theory","hint":string,"marks":number}]}
Mix: ~6 Easy, ~9 Medium, ~5 Hard. Questions must be specific to the actual code.`,
      `Filename: ${req.file.originalname}\nLanguage: ${lang}\n\nCode:\n\`\`\`${lang}\n${code.slice(0, 8000)}\n\`\`\``,
      4096
    );
    log("INFO", "viva_generated", { userId: req.user?.id ?? "anon", lang });
    res.json({ success: true, data: parseAIJson(raw) });
  } catch (e) { next(e); }
});

const debugSchema = Joi.object({
  code:     Joi.string().trim().min(2).max(10000).required(),
  language: Joi.string().trim().max(50).default("python"),
  error:    Joi.string().trim().max(2000).allow("").default(""),
});

app.post("/api/debug", async (req, res, next) => {
  try {
    const { code, language, error: errMsg } = validate(debugSchema, req.body);
    log("INFO", "debug_request", { userId: req.user?.id ?? "anon", language });
    const raw = await callBedrock(
      `You are a strict Socratic coding mentor. Help students understand bugs — NEVER fix the code.
RULE: Never provide corrected code or complete solutions. Hints and concepts only.
Respond ONLY with valid JSON (no markdown).
Schema: {"language":string,"errorType":string,"errorSummary":string,"conceptExplanation":string,"hints":["hint1","hint2","hint3"],"affectedLines":number[],"difficulty":"Easy"|"Medium"|"Hard","commonMistake":string,"furtherReading":string}`,
      `Language: ${language}\nCode:\n\`\`\`${language}\n${code}\n\`\`\`${errMsg ? `\nError: ${errMsg}` : ""}`,
      2048
    );
    res.json({ success: true, data: parseAIJson(raw) });
  } catch (e) { next(e); }
});

// ─── 404 + Error Handler ──────────────────────────────────────────────────────
app.use((req, _res, next) =>
  next(appError(404, `Route not found: ${req.method} ${req.originalUrl}`, "NOT_FOUND"))
);

// eslint-disable-next-line no-unused-vars
app.use((err, req, res, _next) => {
  const status = err.status ?? 500;
  log(status >= 500 ? "ERROR" : "WARN", "request_error", {
    method: req.method, url: req.url, status, message: err.message, requestId: req.id,
  });
  res.status(status).json({
    success: false,
    error: { code: err.code ?? "SERVER_ERROR", message: (err.isOp || !IS_PROD) ? err.message : "An unexpected error occurred" },
    meta:  { requestId: req.id, ts: new Date().toISOString() },
  });
});

// ─── Start ────────────────────────────────────────────────────────────────────
const server = app.listen(PORT, async () => {
  if (process.env.AWS_ACCESS_KEY_ID)
    _ensureLogStream().catch(e => console.warn("[CW] Log stream init failed:", e.message));

  const ok = "✅", warn = "⚠️ ";
  console.log(`\n🚀  BharatLearn API  →  http://localhost:${PORT}`);
  console.log(`    Google OAuth   : ${GOOGLE_CLIENT_ID           ? ok : warn} ${GOOGLE_CLIENT_ID           ? "configured"    : "not configured"}`);
  console.log(`    Groq AI  : ${process.env.GROQ_API_KEY ? ok : warn} ${process.env.GROQ_API_KEY ? "configured" : "not configured"}`);
  console.log(`    Model    : ${ok} ${GROQ_MODEL}`);
  console.log(`    DynamoDB       : ${process.env.AWS_ACCESS_KEY_ID ? ok : warn} ${process.env.AWS_ACCESS_KEY_ID ? "ready"       : "needs AWS keys"}`);
  console.log(`    S3 Bucket      : ${ok} ${S3_BUCKET}`);
  console.log(`    CloudWatch     : ${process.env.AWS_ACCESS_KEY_ID ? ok : warn} ${process.env.AWS_ACCESS_KEY_ID ? CW_LOG_GROUP  : "needs AWS keys"}\n`);
  log("INFO", "server_started", { port: PORT, env: process.env.NODE_ENV });
});

["SIGTERM","SIGINT"].forEach(sig =>
  process.on(sig, () => {
    log("INFO", "server_shutdown", { signal: sig });
    clearInterval(_cwFlushTimer);
    _flushCWLogs().finally(() => server.close(() => { console.log("Server closed."); process.exit(0); }));
  })
);
process.on("unhandledRejection", r => { log("ERROR", "unhandled_rejection", { reason: String(r) }); process.exit(1); });

export default server;
