import scrapy
import logging

class AllsidesSpider(scrapy.Spider):
    name = 'Allsides_spider'
    #start_urls = ['http://brickset.com/sets/year-2016', 'https://www.allsides.com/story/admin', "https://www.allsides.com/unbiased-balanced-news"]
    start_urls = ["https://www.allsides.com/unbiased-balanced-news"]
    def parse(self, response):
        logging.info("I am Parsing a response")
        logging.info(response)
        SET_SELECTOR = '.even'
        SET_SELECTOR = '.odd'
        for SET_SELECTOR in ['.even', '.odd', '.set']:
            for items in response.css(SET_SELECTOR):

                NAME_SELECTOR = 'td ::text'
                LINK_SELECTOR = 'td a ::attr(href)'
                PIECES_SELECTOR = './/dl[dt/text() = "Pieces"]/dd/a/text()'
                MINIFIGS_SELECTOR = './/dl[dt/text() = "Minifigs"]/dd[2]/a/text()'
                IMAGE_SELECTOR = 'img ::attr(src)'
                yield {
                    'name': items.css(NAME_SELECTOR).extract_first(),
                    'link': items.css(LINK_SELECTOR).extract_first(),
                    'pieces': items.xpath(PIECES_SELECTOR).extract_first(),
                    #'minifigs': brickset.xpath(MINIFIGS_SELECTOR).extract_first(),
                    #'image': brickset.css(IMAGE_SELECTOR).extract_first(),
                }

        """NEXT_PAGE_SELECTOR = '.next a ::attr(href)'
        next_page = response.css(NEXT_PAGE_SELECTOR).extract_first()
        if next_page:
            yield scrapy.Request(
                response.urljoin(next_page),
                callback=self.parse
            )
        """