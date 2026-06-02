// Dragon Engine — Axios client
import axios from 'axios'

const client = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 600000, // 10 min — pipeline can take time (ths_hot especially)
  headers: { 'Content-Type': 'application/json' },
})

client.interceptors.response.use(
  (res) => res,
  (err) => {
    console.error('[API]', err.config?.url, err.message)
    return Promise.reject(err)
  },
)

export default client
