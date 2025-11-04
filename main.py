import telebot
import pandas as pd
import feedparser
import requests
from datetime import datetime
import time
import threading
import os

TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
bot = telebot.TeleBot(TOKEN)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.nseindia.com/"
})

# १. FII/DII
def get_fiidii():
    try:
        date_str = datetime.now().strftime('%d-%b-%Y').upper()
        url = f"https://www.nseindia.com/archives/equities/fiidii/{date_str}.csv"
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            return "❌ FII/DII डेटा अजून नाही."
        df = pd.read_csv(resp.text.splitlines())
        row = df.iloc[0]
        return f"💰 FII: *₹{row['FII Net (Cr)']:,.0f} Cr* | 🏦 DII: *₹{row['DII Net (Cr)']:,.0f} Cr*"
    except: return "❌ FII/DII Error"

# २. Sensex & Nifty (NSE JSON API - 100% रियल!)
def get_indices():
    try:
        url = "https://www.nseindia.com/api/quote-equity?symbol=%5ENSEI"  # Sensex
        data = session.get("https://www.nseindia.com/", timeout=10)  # cookie
        j = session.get(url).json()
        sensex = j['priceInfo']['lastPrice']
        sensex_chg = j['priceInfo']['change']
        sensex_pchg = j['priceInfo']['pChange']

        url2 = "https://www.nseindia.com/api/quote-equity?symbol=%5ENIFTY%2050"
        j2 = session.get(url2).json()
        nifty = j2['priceInfo']['lastPrice']
        nifty_chg = j2['priceInfo']['change']
        nifty_pchg = j2['priceInfo']['pChange']

        return (f"📈 *Sensex*: {sensex:,.0f} ({sensex_chg:+.0f} | {sensex_pchg:+.2f}%)\n"
                f"📊 *Nifty*: {nifty:,.0f} ({nifty_chg:+.0f} | {nifty_pchg:+.2f}%)")
    except:
        return "❌ Indices डेटा लोड होत नाही (मार्केट बंद असेल)"

# ३. न्यूज
def get_news():
    feed = feedparser.parse("https://www.moneycontrol.com/news/rss")
    msg = "📰 *टॉप ३ Sensex/Nifty न्यूज*\n\n"
    for i, entry in enumerate(feed.entries[:3]):
        msg += f"{i+1}. {entry.title}\n🔗 {entry.link}\n\n"
    return msg

# कमांड्स
@bot.message_handler(commands=['start'])
def start(m):
    bot.reply_to(m, "🚀 नवीन अपडेट!\n/fiidii | /sensex | /nifty | /news")

@bot.message_handler(commands=['fiidii'])
def fiidii(m): bot.reply_to(m, get_fiidii(), parse_mode='Markdown')

@bot.message_handler(commands=['sensex'])
def sensex(m): bot.reply_to(m, get_indices(), parse_mode='Markdown')

@bot.message_handler(commands=['nifty'])
def nifty(m): bot.reply_to(m, get_indices(), parse_mode='Markdown')

@bot.message_handler(commands=['news'])
def news(m): bot.reply_to(m, get_news())

# रोज 7:35 PM ऑटो मेसेज
def daily_report():
    while True:
        now = datetime.now()
        if now.hour == 19 and now.minute == 35:
            msg = f"🌙 *आजचा मार्केट अपडेट* ({now.strftime('%d %b')})\n\n"
            msg += get_fiidii() + "\n\n"
            msg += get_indices() + "\n\n"
            msg += get_news()
            bot.send_message(CHAT_ID, msg, parse_mode='Markdown', disable_web_page_preview=True)
            time.sleep(70)
        time.sleep(30)

if __name__ == "__main__":
    print("Bot व्हॉट्सअॅप सारखा चालू! 🚀")
    threading.Thread(target=daily_report, daemon=True).start()
    bot.infinity_polling()
