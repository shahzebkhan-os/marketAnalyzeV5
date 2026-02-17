import asyncio
import json
import aiohttp
from client import GrowwWebProvider

async def debug_raw_json():
    async with aiohttp.ClientSession() as session:
        provider = GrowwWebProvider(session)
        url = "https://groww.in/options/nifty"
        print(f"Fetching {url}...")
        async with session.get(url, headers=provider.headers) as resp:
            html = await resp.text()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            script_tag = soup.find('script', id='__NEXT_DATA__')
            data = json.loads(script_tag.string)
            
            # Navigate to option contracts
            page_props_data = data['props']['pageProps']['data']
            oc_data = page_props_data['optionChain']
            contracts = oc_data['optionContracts']
            
            if contracts:
                sample = contracts[len(contracts)//2] # Middle of chain
                print("--- Sample Contract Raw JSON ---")
                print(json.dumps(sample, indent=2))
            else:
                print("No contracts found")

if __name__ == "__main__":
    asyncio.run(debug_raw_json())
LineContent: 22 
