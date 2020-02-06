SELECT
    Document,
    Word,
    COUNT(1) AS WordCount
FROM
    [dbo].[KeywordsofDocuments]
GROUP BY
    Document, Word