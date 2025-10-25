const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

type RequestOptions = RequestInit & { skipJson?: boolean };

const handleResponse = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    const contentType = response.headers.get('content-type');
    let errorMessage = `Request failed with status ${response.status}`;
    if (contentType && contentType.includes('application/json')) {
      try {
        const data = await response.json();
        if (data && data.detail) {
          errorMessage = Array.isArray(data.detail)
            ? data.detail.map((item: any) => item.msg || item).join('; ')
            : data.detail;
        }
      } catch (err) {
        // ignore parse errors and fall back to generic message
      }
    } else {
      const text = await response.text();
      if (text) {
        errorMessage = text;
      }
    }
    throw new Error(errorMessage);
  }

  if (response.status === 204) {
    return undefined as unknown as T;
  }

  const contentType = response.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    return response.json() as Promise<T>;
  }

  const text = await response.text();
  return text as unknown as T;
};

export const apiClient = {
  async get<T>(path: string, options?: RequestOptions): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: 'GET',
      ...options,
    });
    return handleResponse<T>(response);
  },
  async post<T>(path: string, body?: any, options?: RequestOptions): Promise<T> {
    const init: RequestInit = {
      method: 'POST',
      ...options,
    };

    if (body instanceof FormData) {
      init.body = body;
    } else if (options?.headers?.['Content-Type']) {
      init.body = body;
    } else if (body !== undefined) {
      init.headers = {
        'Content-Type': 'application/json',
        ...(options?.headers || {}),
      };
      init.body = JSON.stringify(body);
    }

    const response = await fetch(`${API_BASE_URL}${path}`, init);
    return handleResponse<T>(response);
  },
};

export default apiClient;
