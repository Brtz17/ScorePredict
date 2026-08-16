import Alpine from 'alpinejs'
import { Client, Databases, Query } from 'appwrite'

window.Alpine = Alpine

const ENDPOINT = 'https://fra.cloud.appwrite.io/v1'
const PROJECT_ID = '6a4927d000138dc9fca2'
const DATABASE_ID = '6a63ee9f00107891d1d5'
const PREDICTIONS_COLLECTION_ID = 'live_predictions'

const client = new Client().setEndpoint(ENDPOINT).setProject(PROJECT_ID)
const databases = new Databases(client)

const FALLBACK_MATCHES = [
  { $id: 'demo-1', competition_code: 'PL', home_team_name: 'Arsenal', away_team_name: 'Liverpool', home_team_crest: 'https://media.api-sports.io/football/teams/42.png', away_team_crest: 'https://media.api-sports.io/football/teams/40.png', exp_home_goals: 1.9, exp_away_goals: 1.3, home_win_prob: 0.52, draw_prob: 0.24, away_win_prob: 0.24, utc_date: '2025-01-15T20:00:00Z' },
  { $id: 'demo-2', competition_code: 'PL', home_team_name: 'Manchester City', away_team_name: 'Manchester United', home_team_crest: 'https://media.api-sports.io/football/teams/50.png', away_team_crest: 'https://media.api-sports.io/football/teams/33.png', exp_home_goals: 2.1, exp_away_goals: 0.9, home_win_prob: 0.63, draw_prob: 0.21, away_win_prob: 0.16, utc_date: '2025-01-16T19:30:00Z' },
  { $id: 'demo-3', competition_code: 'PD', home_team_name: 'Real Madrid', away_team_name: 'Sevilla', home_team_crest: 'https://media.api-sports.io/football/teams/541.png', away_team_crest: 'https://media.api-sports.io/football/teams/536.png', exp_home_goals: 1.2, exp_away_goals: 1.6, home_win_prob: 0.29, draw_prob: 0.27, away_win_prob: 0.44, utc_date: '2025-01-17T21:00:00Z' },
]

function initials(name) {
  return (name || '?')
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0])
    .join('')
    .toUpperCase()
}

function tiltFor(idx) {
  const pattern = [-2.4, 1.6, -1.1, 2.2, -1.8, 1.1]
  return pattern[idx % pattern.length] + 'deg'
}

function tapeTiltFor(idx) {
  const pattern = [-7, 5, -4, 8, -6]
  return pattern[idx % pattern.length] + 'deg'
}

function formatMatchDate(utcDate) {
  if (!utcDate) return 'TBD'
  
  try {
    const date = new Date(utcDate)
    
    const day = String(date.getUTCDate()).padStart(2, '0')
    const month = String(date.getUTCMonth() + 1).padStart(2, '0')
    const year = date.getUTCFullYear()
    const hours = String(date.getUTCHours()).padStart(2, '0')
    const minutes = String(date.getUTCMinutes()).padStart(2, '0')
    
    return `${day}.${month}.${year}. ${hours}:${minutes}`
  } catch (e) {
    return utcDate
  }
}

const COMPETITION_NAMES = {
  PL: 'Premier League',
  PD: 'La Liga',
  BL1: 'Bundesliga',
  SA: 'Serie A',
  FL1: 'Ligue 1',
  DED: 'Eredivisie',
  PPL: 'Primeira Liga',
  ELC: 'Championship',
  CL: 'Champions League',
}

function competitionName(code) {
  return COMPETITION_NAMES[code] || code || ''
}

const COMPETITION_COLORS = {
  CL: '#1A1F71',
  PL: '#3D195B',
  PD: '#EE8707',
  BL1: '#D3010C',
  SA: '#008FD7',
  FL1: '#0A5FA8',
  DED: '#FF6C00',
  PPL: '#00A650',
  ELC: '#732582',
}

function competitionColor(code) {
  return COMPETITION_COLORS[code] || '#ABDF75'
}

const COMPETITION_ORDER = ['CL', 'PL', 'PD', 'BL1', 'SA', 'FL1', 'DED', 'PPL', 'ELC']

Alpine.store('predictions', {
  matches: [],
  loading: true,
  usingFallback: false,

  async init() {
    try {
      const all = []
      const PAGE_SIZE = 100
      let cursor = null

      while (true) {
        const queries = [Query.orderAsc('utc_date'), Query.limit(PAGE_SIZE)]
        if (cursor) queries.push(Query.cursorAfter(cursor))

        const res = await databases.listDocuments(DATABASE_ID, PREDICTIONS_COLLECTION_ID, queries)
        all.push(...res.documents)

        if (res.documents.length < PAGE_SIZE) break
        cursor = res.documents[res.documents.length - 1].$id
      }

      this.matches = all.length ? all : FALLBACK_MATCHES
      this.usingFallback = all.length === 0
    } catch (err) {
      console.warn('Nem sikerult elerni az Appwrite predikciokat, demo-adat jelenik meg.', err)
      this.matches = FALLBACK_MATCHES
      this.usingFallback = true
    } finally {
      this.loading = false
    }
  },

  byId(id) {
    return this.matches.find((m) => m.$id === id)
  },

  get groupedMatches() {
    const groups = {}
    for (const m of this.matches) {
      const code = m.competition_code || 'OTHER'
      if (!groups[code]) groups[code] = []
      groups[code].push(m)
    }

    const knownCodes = COMPETITION_ORDER.filter((code) => groups[code])
    const otherCodes = Object.keys(groups).filter((code) => !COMPETITION_ORDER.includes(code))

    return [...knownCodes, ...otherCodes].map((code) => ({
      code,
      name: competitionName(code),
      color: competitionColor(code),
      matches: groups[code],
    }))
  },

  initials,
  tiltFor,
  tapeTiltFor,
  competitionName,
  competitionColor,
  formatMatchDate,
})

Alpine.start()