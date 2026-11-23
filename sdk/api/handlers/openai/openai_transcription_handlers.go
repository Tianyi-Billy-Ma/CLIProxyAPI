package openai

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/MoeForge/moego/ojson"
	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/clienterror"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
	log "github.com/sirupsen/logrus"
)

const (
	openAITranscriptionHandlerType = "openai-transcription"
	openAITranscriptionPath        = "/v1/audio/transcriptions"
	transcriptionJobTTL            = 24 * time.Hour
	// maxActiveTranscriptionJobs bounds async uploads held in memory at once.
	maxActiveTranscriptionJobs = 32
)

// AudioTranscriptions proxies an OpenAI multipart transcription request to an
// OpenAI-compatible upstream. With "Prefer: respond-async" it answers 202 with a
// job id that is polled via AudioTranscriptionRetrieve.
func (h *OpenAIAPIHandler) AudioTranscriptions(c *gin.Context) {
	raw, err := handlers.ReadRequestBody(c)
	if err != nil {
		writeSpeechError(c, clienterror.HTTPStatusFromErrorOr(err, http.StatusBadRequest), fmt.Sprintf("Invalid request: %v", err))
		return
	}
	contentType := c.GetHeader("Content-Type")
	model, err := transcriptionModelFromMultipart(raw, contentType)
	if err != nil {
		writeSpeechError(c, http.StatusBadRequest, err.Error())
		return
	}
	headers := c.Request.Header.Clone()
	headers.Del("Content-Encoding")

	if !preferRespondAsync(c.Request.Header) {
		cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
		resp, errMsg := h.executeTranscription(cliCtx, model, raw, headers)
		if errMsg != nil {
			h.WriteErrorResponse(c, errMsg)
			cliCancel(errMsg.Error)
			return
		}
		handlers.WriteUpstreamHeaders(c.Writer.Header(), resp.Headers)
		c.Writer.Header().Del("Content-Length")
		if c.Writer.Header().Get("Content-Type") == "" {
			if ojson.ValidBytes(resp.Body) {
				c.Header("Content-Type", "application/json")
			} else {
				c.Header("Content-Type", "text/plain; charset=utf-8")
			}
		}
		c.Status(http.StatusOK)
		_, _ = c.Writer.Write(resp.Body)
		cliCancel(nil)
		return
	}

	job, ok := transcriptionJobs.create(c.GetString("userApiKey"))
	if !ok {
		c.JSON(http.StatusTooManyRequests, handlers.ErrorResponse{Error: handlers.ErrorDetail{
			Message: fmt.Sprintf("too many transcription jobs in progress (limit %d)", maxActiveTranscriptionJobs),
			Type:    "rate_limit_error",
			Code:    "too_many_jobs",
		}})
		return
	}
	// The job outlives the request: keep request-scoped values (usage identity)
	// but neither its cancellation nor the pooled gin context.
	jobCtx := context.WithoutCancel(c.Request.Context())
	go func() {
		transcriptionJobs.setStatus(job.id, "processing")
		resp, errMsg := h.executeTranscription(jobCtx, model, raw, headers)
		if errMsg != nil {
			transcriptionJobs.fail(job.id, errMsg)
			return
		}
		transcriptionJobs.complete(job.id, resp.Body)
	}()
	c.JSON(http.StatusAccepted, gin.H{"id": job.id, "status": "queued"})
}

// AudioTranscriptionRetrieve returns the state of an async transcription job.
func (h *OpenAIAPIHandler) AudioTranscriptionRetrieve(c *gin.Context) {
	body, ok := transcriptionJobs.render(c.Param("id"), c.GetString("userApiKey"))
	if !ok {
		c.JSON(http.StatusNotFound, handlers.ErrorResponse{Error: handlers.ErrorDetail{
			Message: "transcription job not found",
			Type:    "invalid_request_error",
			Code:    "not_found",
		}})
		return
	}
	c.Data(http.StatusOK, "application/json", body)
}

func (h *OpenAIAPIHandler) executeTranscription(ctx context.Context, model string, body []byte, headers http.Header) (handlers.ModelExecutionResponse, *interfaces.ErrorMessage) {
	return h.ExecuteProtocolWithAuthManager(ctx, handlers.ProtocolExecutionRequest{
		EntryProtocol: openAITranscriptionHandlerType,
		ExitProtocol:  openAITranscriptionHandlerType,
		Model:         model,
		Body:          body,
		Headers:       headers,
		Path:          openAITranscriptionPath,
	})
}

func preferRespondAsync(header http.Header) bool {
	for _, value := range header.Values("Prefer") {
		for _, token := range strings.Split(value, ",") {
			if strings.EqualFold(strings.TrimSpace(token), "respond-async") {
				return true
			}
		}
	}
	return false
}

// transcriptionModelFromMultipart validates the form and returns its model field.
func transcriptionModelFromMultipart(raw []byte, contentType string) (string, error) {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil || !strings.EqualFold(mediaType, "multipart/form-data") {
		return "", errors.New("Content-Type must be multipart/form-data")
	}
	boundary := params["boundary"]
	if boundary == "" {
		return "", errors.New("multipart boundary is missing")
	}
	reader := multipart.NewReader(bytes.NewReader(raw), boundary)
	model, hasFile := "", false
	for {
		part, errPart := reader.NextPart()
		if errPart == io.EOF {
			break
		}
		if errPart != nil {
			return "", fmt.Errorf("invalid multipart body: %w", errPart)
		}
		switch part.FormName() {
		case "file":
			hasFile = part.FileName() != ""
		case "model":
			value, errRead := io.ReadAll(io.LimitReader(part, 1024))
			if errRead != nil {
				return "", fmt.Errorf("invalid multipart body: %w", errRead)
			}
			model = strings.TrimSpace(string(value))
		}
		_ = part.Close()
	}
	if !hasFile {
		return "", errors.New("file is required")
	}
	if model == "" {
		return "", errors.New("model is required")
	}
	return model, nil
}

type transcriptionJob struct {
	id       string
	owner    string
	status   string
	result   []byte
	errMsg   string
	errCode  string
	finished time.Time
}

// transcriptionJobStore keeps async jobs in memory; finished jobs expire after
// transcriptionJobTTL and are only visible to the API key that created them.
// shortcut: jobs are lost on restart and expired ones are swept lazily on
// create/poll; persist them if restarts mid-job become a problem.
type transcriptionJobStore struct {
	mu   sync.Mutex
	jobs map[string]*transcriptionJob
	now  func() time.Time
}

var transcriptionJobs = &transcriptionJobStore{jobs: map[string]*transcriptionJob{}, now: time.Now}

func (s *transcriptionJobStore) create(owner string) (*transcriptionJob, bool) {
	var buf [16]byte
	if _, err := rand.Read(buf[:]); err != nil {
		log.Errorf("transcription: random job id: %v", err)
		return nil, false
	}
	job := &transcriptionJob{id: "transcr_" + hex.EncodeToString(buf[:]), owner: owner, status: "queued"}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.evictLocked()
	active := 0
	for _, existing := range s.jobs {
		if existing.finished.IsZero() {
			active++
		}
	}
	if active >= maxActiveTranscriptionJobs {
		return nil, false
	}
	s.jobs[job.id] = job
	return job, true
}

func (s *transcriptionJobStore) setStatus(id, status string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if job := s.jobs[id]; job != nil {
		job.status = status
	}
}

func (s *transcriptionJobStore) complete(id string, result []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if job := s.jobs[id]; job != nil {
		job.status, job.result, job.finished = "completed", result, s.now()
	}
}

func (s *transcriptionJobStore) fail(id string, errMsg *interfaces.ErrorMessage) {
	message, code := "transcription failed", "upstream_error"
	if errMsg != nil {
		if errMsg.StatusCode > 0 {
			code = strconv.Itoa(errMsg.StatusCode)
		}
		if errMsg.Error != nil {
			message = errMsg.Error.Error()
			// Keep the upstream OpenAI error envelope's own message and code.
			if upstream := []byte(message); ojson.ValidBytes(upstream) {
				if m := ojson.GetBytes(upstream, "error.message").String(); m != "" {
					message = m
				}
				if cv := ojson.GetBytes(upstream, "error.code").String(); cv != "" {
					code = cv
				}
			}
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if job := s.jobs[id]; job != nil {
		job.status, job.errMsg, job.errCode, job.finished = "failed", message, code, s.now()
	}
}

func (s *transcriptionJobStore) evictLocked() {
	now := s.now()
	for id, job := range s.jobs {
		if !job.finished.IsZero() && now.Sub(job.finished) > transcriptionJobTTL {
			delete(s.jobs, id)
		}
	}
}

// render returns the poll response body for a job, or false when it is unknown or expired.
func (s *transcriptionJobStore) render(id, owner string) ([]byte, bool) {
	s.mu.Lock()
	s.evictLocked()
	stored := s.jobs[id]
	var job transcriptionJob
	if stored != nil {
		job = *stored
	}
	s.mu.Unlock()
	if stored == nil || job.owner != owner {
		return nil, false
	}
	var doc *ojson.Doc
	switch job.status {
	case "completed":
		if ojson.ValidBytes(job.result) && ojson.ParseBytes(job.result).IsObject() {
			doc = ojson.NewDoc(job.result)
		} else {
			// Non-JSON formats (text, srt, vtt) are returned as text.
			doc = ojson.NewDocString(`{}`)
			_ = doc.Set("text", string(job.result))
		}
	case "failed":
		doc = ojson.NewDocString(`{}`)
		_ = doc.Set("error.message", job.errMsg)
		_ = doc.Set("error.code", job.errCode)
	default:
		doc = ojson.NewDocString(`{}`)
	}
	_ = doc.Set("id", job.id)
	_ = doc.Set("status", job.status)
	return doc.Bytes(), true
}
